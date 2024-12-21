// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of GridFunction

#include "gridfunc.hpp"
#include "linearform.hpp"
#include "bilinearform.hpp"
#include "quadinterpolator.hpp"
#include "../mesh/nurbs.hpp"
#include "../general/text.hpp"

#ifdef MFEM_USE_MPI
#include "pfespace.hpp"
#endif

#include <limits>
#include <cstring>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace mfem
{

using namespace std;

GridFunction::GridFunction(Mesh *m, std::istream &input)
   : Vector()
{
   // Grid functions are stored on the device
   UseDevice(true);

   fes = new FiniteElementSpace;
   fec_owned = fes->Load(m, input);

   skip_comment_lines(input, '#');
   istream::int_type next_char = input.peek();
   if (next_char == 'N') // First letter of "NURBS_patches"
   {
      string buff;
      getline(input, buff);
      filter_dos(buff);
      if (buff == "NURBS_patches")
      {
         MFEM_VERIFY(fes->GetNURBSext(),
                     "NURBS_patches requires NURBS FE space");
         fes->GetNURBSext()->LoadSolution(input, *this);
      }
      else
      {
         MFEM_ABORT("unknown section: " << buff);
      }
   }
   else
   {
      Vector::Load(input, fes->GetVSize());

      // if the mesh is a legacy (v1.1) NC mesh, it has old vertex ordering
      if (fes->Nonconforming() &&
          fes->GetMesh()->ncmesh->IsLegacyLoaded())
      {
         LegacyNCReorder();
      }
   }
   fes_sequence = fes->GetSequence();
}

GridFunction::GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces)
{
   UseDevice(true);

   // all GridFunctions must have the same FE collection, vdim, ordering
   int vdim, ordering;

   fes = gf_array[0]->FESpace();
   fec_owned = FiniteElementCollection::New(fes->FEColl()->Name());
   vdim = fes->GetVDim();
   ordering = fes->GetOrdering();
   fes = new FiniteElementSpace(m, fec_owned, vdim, ordering);
   SetSize(fes->GetVSize());

   if (m->NURBSext)
   {
      m->NURBSext->MergeGridFunctions(gf_array, num_pieces, *this);
      return;
   }

   int g_ndofs  = fes->GetNDofs();
   int g_nvdofs = fes->GetNVDofs();
   int g_nedofs = fes->GetNEDofs();
   int g_nfdofs = fes->GetNFDofs();
   int g_nddofs = g_ndofs - (g_nvdofs + g_nedofs + g_nfdofs);
   int vi, ei, fi, di;
   vi = ei = fi = di = 0;
   for (int i = 0; i < num_pieces; i++)
   {
      FiniteElementSpace *l_fes = gf_array[i]->FESpace();
      int l_ndofs  = l_fes->GetNDofs();
      int l_nvdofs = l_fes->GetNVDofs();
      int l_nedofs = l_fes->GetNEDofs();
      int l_nfdofs = l_fes->GetNFDofs();
      int l_nddofs = l_ndofs - (l_nvdofs + l_nedofs + l_nfdofs);
      const real_t *l_data = gf_array[i]->GetData();
      real_t *g_data = data;
      if (ordering == Ordering::byNODES)
      {
         for (int d = 0; d < vdim; d++)
         {
            memcpy(g_data+vi, l_data, l_nvdofs*sizeof(real_t));
            l_data += l_nvdofs;
            g_data += g_nvdofs;
            memcpy(g_data+ei, l_data, l_nedofs*sizeof(real_t));
            l_data += l_nedofs;
            g_data += g_nedofs;
            memcpy(g_data+fi, l_data, l_nfdofs*sizeof(real_t));
            l_data += l_nfdofs;
            g_data += g_nfdofs;
            memcpy(g_data+di, l_data, l_nddofs*sizeof(real_t));
            l_data += l_nddofs;
            g_data += g_nddofs;
         }
      }
      else
      {
         memcpy(g_data+vdim*vi, l_data, l_nvdofs*sizeof(real_t)*vdim);
         l_data += vdim*l_nvdofs;
         g_data += vdim*g_nvdofs;
         memcpy(g_data+vdim*ei, l_data, l_nedofs*sizeof(real_t)*vdim);
         l_data += vdim*l_nedofs;
         g_data += vdim*g_nedofs;
         memcpy(g_data+vdim*fi, l_data, l_nfdofs*sizeof(real_t)*vdim);
         l_data += vdim*l_nfdofs;
         g_data += vdim*g_nfdofs;
         memcpy(g_data+vdim*di, l_data, l_nddofs*sizeof(real_t)*vdim);
         l_data += vdim*l_nddofs;
         g_data += vdim*g_nddofs;
      }
      vi += l_nvdofs;
      ei += l_nedofs;
      fi += l_nfdofs;
      di += l_nddofs;
   }
   fes_sequence = fes->GetSequence();
}

void GridFunction::Destroy()
{
   if (fec_owned)
   {
      delete fes;
      delete fec_owned;
      fec_owned = NULL;
   }
}

void GridFunction::Update()
{
   if (fes->GetSequence() == fes_sequence)
   {
      return; // space and grid function are in sync, no-op
   }
   // it seems we cannot use the following, due to FESpace::Update(false)
   /*if (fes->GetSequence() != fes_sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. GridFunction needs to be updated "
                 "right after the space is updated.");
   }*/
   fes_sequence = fes->GetSequence();

   const Operator *T = fes->GetUpdateOperator();
   if (T)
   {
      Vector old_data;
      old_data.Swap(*this);
      SetSize(T->Height());
      UseDevice(true);
      T->Mult(old_data, *this);
   }
   else
   {
      SetSize(fes->GetVSize());
   }

   if (t_vec.Size() > 0) { SetTrueVector(); }
}

void GridFunction::SetSpace(FiniteElementSpace *f)
{
   if (f != fes) { Destroy(); }
   fes = f;
   SetSize(fes->GetVSize());
   fes_sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, real_t *v)
{
   if (f != fes) { Destroy(); }
   fes = f;
   NewDataAndSize(v, fes->GetVSize());
   fes_sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   MFEM_ASSERT(v.Size() >= v_offset + f->GetVSize(), "");
   if (f != fes) { Destroy(); }
   fes = f;
   v.UseDevice(true);
   this->Vector::MakeRef(v, v_offset, fes->GetVSize());
   fes_sequence = fes->GetSequence();
}

void GridFunction::MakeTRef(FiniteElementSpace *f, real_t *tv)
{
   if (IsIdentityProlongation(f->GetProlongationMatrix()))
   {
      MakeRef(f, tv);
      t_vec.NewDataAndSize(tv, size);
   }
   else
   {
      SetSpace(f); // works in parallel
      t_vec.NewDataAndSize(tv, f->GetTrueVSize());
   }
}

void GridFunction::MakeTRef(FiniteElementSpace *f, Vector &tv, int tv_offset)
{
   tv.UseDevice(true);
   if (IsIdentityProlongation(f->GetProlongationMatrix()))
   {
      MakeRef(f, tv, tv_offset);
      t_vec.NewMemoryAndSize(data, size, false);
   }
   else
   {
      MFEM_ASSERT(tv.Size() >= tv_offset + f->GetTrueVSize(), "");
      SetSpace(f); // works in parallel
      t_vec.MakeRef(tv, tv_offset, f->GetTrueVSize());
   }
}

void GridFunction::SumFluxAndCount(BilinearFormIntegrator &blfi,
                                   GridFunction &flux,
                                   Array<int>& count,
                                   bool wcoef,
                                   int subdomain)
{
   GridFunction &u = *this;

   ElementTransformation *Transf;
   DofTransformation *udoftrans;
   DofTransformation *fdoftrans;

   FiniteElementSpace *ufes = u.FESpace();
   FiniteElementSpace *ffes = flux.FESpace();

   int nfe = ufes->GetNE();
   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl;

   flux = 0.0;
   count = 0;

   for (int i = 0; i < nfe; i++)
   {
      if (subdomain >= 0 && ufes->GetAttribute(i) != subdomain)
      {
         continue;
      }

      udoftrans = ufes->GetElementVDofs(i, udofs);
      fdoftrans = ffes->GetElementVDofs(i, fdofs);

      u.GetSubVector(udofs, ul);
      if (udoftrans)
      {
         udoftrans->InvTransformPrimal(ul);
      }

      Transf = ufes->GetElementTransformation(i);
      blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                              *ffes->GetFE(i), fl, wcoef);

      if (fdoftrans)
      {
         fdoftrans->TransformPrimal(fl);
      }
      flux.AddElementVector(fdofs, fl);

      FiniteElementSpace::AdjustVDofs(fdofs);
      for (int j = 0; j < fdofs.Size(); j++)
      {
         count[fdofs[j]]++;
      }
   }
}

void GridFunction::ComputeFlux(BilinearFormIntegrator &blfi,
                               GridFunction &flux, bool wcoef,
                               int subdomain)
{
   Array<int> count(flux.Size());

   SumFluxAndCount(blfi, flux, count, wcoef, subdomain);

   // complete averaging
   for (int i = 0; i < count.Size(); i++)
   {
      if (count[i] != 0) { flux(i) /= count[i]; }
   }
}

int GridFunction::VectorDim() const
{
   const FiniteElement *fe;
   if (!fes->GetNE())
   {
      static const Geometry::Type geoms[3] =
      { Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::TETRAHEDRON };
      fe = fes->FEColl()->
           FiniteElementForGeometry(geoms[fes->GetMesh()->Dimension()-1]);
   }
   else
   {
      fe = fes->GetFE(0);
   }
   if (!fe || fe->GetRangeType() == FiniteElement::SCALAR)
   {
      return fes->GetVDim();
   }
   return fes->GetVDim()*std::max(fes->GetMesh()->SpaceDimension(),
                                  fe->GetRangeDim());
}

int GridFunction::CurlDim() const
{
   const FiniteElement *fe;
   if (!fes->GetNE())
   {
      static const Geometry::Type geoms[3] =
      { Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::TETRAHEDRON };
      fe = fes->FEColl()->
           FiniteElementForGeometry(geoms[fes->GetMesh()->Dimension()-1]);
   }
   else
   {
      fe = fes->GetFE(0);
   }
   if (!fe || fe->GetRangeType() == FiniteElement::SCALAR)
   {
      return 2 * fes->GetMesh()->SpaceDimension() - 3;
   }
   return fes->GetVDim()*fe->GetCurlDim();
}

void GridFunction::GetTrueDofs(Vector &tv) const
{
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R || IsIdentityProlongation(fes->GetProlongationMatrix()))
   {
      // R is identity
      tv = *this; // no real copy if 'tv' and '*this' use the same data
   }
   else
   {
      tv.SetSize(R->Height());
      R->Mult(*this, tv);
   }
}

void GridFunction::SetFromTrueDofs(const Vector &tv)
{
   MFEM_ASSERT(tv.Size() == fes->GetTrueVSize(), "invalid input");
   const SparseMatrix *cP = fes->GetConformingProlongation();
   if (!cP)
   {
      *this = tv; // no real copy if 'tv' and '*this' use the same data
   }
   else
   {
      cP->Mult(tv, *this);
   }
}

void GridFunction::GetNodalValues(int i, Array<real_t> &nval, int vdim) const
{
   Array<int> vdofs;

   DofTransformation * doftrans = fes->GetElementVDofs(i, vdofs);
   const FiniteElement *FElem = fes->GetFE(i);
   const IntegrationRule *ElemVert =
      Geometries.GetVertices(FElem->GetGeomType());
   int dof = FElem->GetDof();
   int n = ElemVert->GetNPoints();
   nval.SetSize(n);
   vdim--;
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }

   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      Vector shape(dof);
      if (FElem->GetMapType() == FiniteElement::VALUE)
      {
         for (int k = 0; k < n; k++)
         {
            FElem->CalcShape(ElemVert->IntPoint(k), shape);
            nval[k] = shape * (&loc_data[dof * vdim]);
         }
      }
      else
      {
         ElementTransformation *Tr = fes->GetElementTransformation(i);
         for (int k = 0; k < n; k++)
         {
            Tr->SetIntPoint(&ElemVert->IntPoint(k));
            FElem->CalcPhysShape(*Tr, shape);
            nval[k] = shape * (&loc_data[dof * vdim]);
         }
      }
   }
   else
   {
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      DenseMatrix vshape(dof, FElem->GetDim());
      for (int k = 0; k < n; k++)
      {
         Tr->SetIntPoint(&ElemVert->IntPoint(k));
         FElem->CalcVShape(*Tr, vshape);
         nval[k] = loc_data * (&vshape(0,vdim));
      }
   }
}

real_t GridFunction::GetValue(int i, const IntegrationPoint &ip, int vdim)
const
{
   Array<int> dofs;
   DofTransformation * doftrans = fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   Vector DofVal(dofs.Size()), LocVec;
   const FiniteElement *fe = fes->GetFE(i);
   if (fe->GetMapType() == FiniteElement::VALUE)
   {
      fe->CalcShape(ip, DofVal);
   }
   else
   {
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      Tr->SetIntPoint(&ip);
      fe->CalcPhysShape(*Tr, DofVal);
   }
   GetSubVector(dofs, LocVec);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(LocVec);
   }

   return (DofVal * LocVec);
}

void GridFunction::GetVectorValue(int i, const IntegrationPoint &ip,
                                  Vector &val) const
{
   const FiniteElement *FElem = fes->GetFE(i);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   DofTransformation * doftrans = fes->GetElementVDofs(i, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      Vector shape(dof);
      if (FElem->GetMapType() == FiniteElement::VALUE)
      {
         FElem->CalcShape(ip, shape);
      }
      else
      {
         ElementTransformation *Tr = fes->GetElementTransformation(i);
         Tr->SetIntPoint(&ip);
         FElem->CalcPhysShape(*Tr, shape);
      }
      int vdim = fes->GetVDim();
      val.SetSize(vdim);
      for (int k = 0; k < vdim; k++)
      {
         val(k) = shape * (&loc_data[dof * k]);
      }
   }
   else
   {
      int vdim = VectorDim();
      DenseMatrix vshape(dof, vdim);
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      Tr->SetIntPoint(&ip);
      FElem->CalcVShape(*Tr, vshape);
      val.SetSize(vdim);
      vshape.MultTranspose(loc_data, val);
   }
}

void GridFunction::GetValues(int i, const IntegrationRule &ir, Vector &vals,
                             int vdim)
const
{
   Array<int> dofs;
   int n = ir.GetNPoints();
   vals.SetSize(n);
   DofTransformation * doftrans = fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   int dof = FElem->GetDof();
   Vector DofVal(dof), loc_data(dof);
   GetSubVector(dofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }
   if (FElem->GetMapType() == FiniteElement::VALUE)
   {
      for (int k = 0; k < n; k++)
      {
         FElem->CalcShape(ir.IntPoint(k), DofVal);
         vals(k) = DofVal * loc_data;
      }
   }
   else
   {
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      for (int k = 0; k < n; k++)
      {
         Tr->SetIntPoint(&ir.IntPoint(k));
         FElem->CalcPhysShape(*Tr, DofVal);
         vals(k) = DofVal * loc_data;
      }
   }
}

void GridFunction::GetValues(int i, const IntegrationRule &ir, Vector &vals,
                             DenseMatrix &tr, int vdim)
const
{
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   ET->Transform(ir, tr);

   GetValues(i, ir, vals, vdim);
}

void GridFunction::GetLaplacians(int i, const IntegrationRule &ir, Vector &laps,
                                 int vdim)
const
{
   Array<int> dofs;
   int n = ir.GetNPoints();
   laps.SetSize(n);
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");

   int dof = FElem->GetDof();
   Vector DofLap(dof), loc_data(dof);
   GetSubVector(dofs, loc_data);
   for (int k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      ET->SetIntPoint(&ip);
      FElem->CalcPhysLaplacian(*ET, DofLap);
      laps(k) = DofLap * loc_data;
   }
}

void GridFunction::GetLaplacians(int i, const IntegrationRule &ir, Vector &laps,
                                 DenseMatrix &tr, int vdim)
const
{
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   ET->Transform(ir, tr);

   GetLaplacians(i, ir, laps, vdim);
}


void GridFunction::GetHessians(int i, const IntegrationRule &ir,
                               DenseMatrix &hess,
                               int vdim)
const
{

   Array<int> dofs;
   int n = ir.GetNPoints();
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   int dim = FElem->GetDim();
   int size = (dim*(dim+1))/2;

   MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");

   int dof = FElem->GetDof();
   DenseMatrix DofHes(dof, size);
   hess.SetSize(n, size);

   Vector loc_data(dof);
   GetSubVector(dofs, loc_data);

   hess = 0.0;
   for (int k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      ET->SetIntPoint(&ip);
      FElem->CalcPhysHessian(*ET, DofHes);

      for (int j = 0; j < size; j++)
      {
         for (int d = 0; d < dof; d++)
         {
            hess(k,j) += DofHes(d,j) * loc_data[d];
         }
      }
   }
}

void GridFunction::GetHessians(int i, const IntegrationRule &ir,
                               DenseMatrix &hess,
                               DenseMatrix &tr, int vdim)
const
{
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   ET->Transform(ir, tr);

   GetHessians(i, ir, hess, vdim);
}


int GridFunction::GetFaceValues(int i, int side, const IntegrationRule &ir,
                                Vector &vals, DenseMatrix &tr,
                                int vdim) const
{
   int n, dir;
   FaceElementTransformations *Transf;

   n = ir.GetNPoints();
   IntegrationRule eir(n);  // ---
   if (side == 2) // automatic choice of side
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 0);
      if (Transf->Elem2No < 0 ||
          fes->GetAttribute(Transf->Elem1No) <=
          fes->GetAttribute(Transf->Elem2No))
      {
         dir = 0;
      }
      else
      {
         dir = 1;
      }
   }
   else
   {
      if (side == 1 && !fes->GetMesh()->FaceIsInterior(i))
      {
         dir = 0;
      }
      else
      {
         dir = side;
      }
   }
   if (dir == 0)
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 4);
      Transf->Loc1.Transform(ir, eir);
      GetValues(Transf->Elem1No, eir, vals, tr, vdim);
   }
   else
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 8);
      Transf->Loc2.Transform(ir, eir);
      GetValues(Transf->Elem2No, eir, vals, tr, vdim);
   }

   return dir;
}

void GridFunction::GetVectorValues(int i, const IntegrationRule &ir,
                                   DenseMatrix &vals, DenseMatrix &tr) const
{
   ElementTransformation *Tr = fes->GetElementTransformation(i);
   Tr->Transform(ir, tr);

   GetVectorValues(*Tr, ir, vals);
}

real_t GridFunction::GetValue(ElementTransformation &T,
                              const IntegrationPoint &ip,
                              int comp, Vector *tr) const
{
   if (tr)
   {
      T.SetIntPoint(&ip);
      T.Transform(ip, *tr);
   }

   const FiniteElement * fe = NULL;
   Array<int> dofs;

   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
         fe = fes->GetFE(T.ElementNo);
         fes->GetElementDofs(T.ElementNo, dofs);
         break;
      case ElementTransformation::EDGE:
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            fe = fes->GetEdgeElement(T.ElementNo);
            fes->GetEdgeDofs(T.ElementNo, dofs);
         }
         else
         {
            MFEM_ABORT("GridFunction::GetValue: Field continuity type \""
                       << fes->FEColl()->GetContType() << "\" not supported "
                       << "on mesh edges.");
            return NAN;
         }
         break;
      case ElementTransformation::FACE:
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            fe = fes->GetFaceElement(T.ElementNo);
            fes->GetFaceDofs(T.ElementNo, dofs);
         }
         else
         {
            MFEM_ABORT("GridFunction::GetValue: Field continuity type \""
                       << fes->FEColl()->GetContType() << "\" not supported "
                       << "on mesh faces.");
            return NAN;
         }
         break;
      case ElementTransformation::BDR_ELEMENT:
      {
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            // This is a continuous field so we can evaluate it on the boundary.
            fe = fes->GetBE(T.ElementNo);
            fes->GetBdrElementDofs(T.ElementNo, dofs);
         }
         else
         {
            // This is a discontinuous field which cannot be evaluated on the
            // boundary so we'll evaluate it in the neighboring element.
            FaceElementTransformations * FET =
               fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);
            MFEM_ASSERT(FET != nullptr,
                        "FaceElementTransformation must be valid for a boundary element");

            // Boundary elements and boundary faces may have different
            // orientations so adjust the integration point if necessary.
            int f, o;
            fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
            IntegrationPoint fip =
               Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o, ip);

            // Compute and set the point in element 1 from fip
            FET->SetAllIntPoints(&fip);
            ElementTransformation & T1 = FET->GetElement1Transformation();
            return GetValue(T1, T1.GetIntPoint(), comp);
         }
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);

         // Evaluate in neighboring element for both continuous and
         // discontinuous fields (the integration point in T1 should have
         // already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         return GetValue(T1, T1.GetIntPoint(), comp);
      }
      default:
      {
         MFEM_ABORT("GridFunction::GetValue: Unsupported element type \""
                    << T.ElementType << "\"");
         return NAN;
      }
   }

   fes->DofsToVDofs(comp-1, dofs);
   Vector DofVal(dofs.Size()), LocVec;
   if (fe->GetMapType() == FiniteElement::VALUE)
   {
      fe->CalcShape(ip, DofVal);
   }
   else
   {
      fe->CalcPhysShape(T, DofVal);
   }
   GetSubVector(dofs, LocVec);

   return (DofVal * LocVec);
}

void GridFunction::GetValues(ElementTransformation &T,
                             const IntegrationRule &ir,
                             Vector &vals, int comp,
                             DenseMatrix *tr) const
{
   if (tr)
   {
      T.Transform(ir, *tr);
   }

   int nip = ir.GetNPoints();
   vals.SetSize(nip);
   for (int j = 0; j < nip; j++)
   {
      const IntegrationPoint &ip = ir.IntPoint(j);
      T.SetIntPoint(&ip);
      vals[j] = GetValue(T, ip, comp);
   }
}

void GridFunction::GetVectorValue(ElementTransformation &T,
                                  const IntegrationPoint &ip,
                                  Vector &val, Vector *tr) const
{
   if (tr)
   {
      T.SetIntPoint(&ip);
      T.Transform(ip, *tr);
   }

   Array<int> vdofs;
   const FiniteElement *fe = NULL;
   DofTransformation * doftrans = NULL;

   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
         doftrans = fes->GetElementVDofs(T.ElementNo, vdofs);
         fe = fes->GetFE(T.ElementNo);
         break;
      case ElementTransformation::EDGE:
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            fe = fes->GetEdgeElement(T.ElementNo);
            fes->GetEdgeVDofs(T.ElementNo, vdofs);
         }
         else
         {
            MFEM_ABORT("GridFunction::GetVectorValue: Field continuity type \""
                       << fes->FEColl()->GetContType() << "\" not supported "
                       << "on mesh edges.");
            return;
         }
         break;
      case ElementTransformation::FACE:
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            fe = fes->GetFaceElement(T.ElementNo);
            fes->GetFaceVDofs(T.ElementNo, vdofs);
         }
         else
         {
            MFEM_ABORT("GridFunction::GetVectorValue: Field continuity type \""
                       << fes->FEColl()->GetContType() << "\" not supported "
                       << "on mesh faces.");
            return;
         }
         break;
      case ElementTransformation::BDR_ELEMENT:
      {
         if (fes->FEColl()->GetContType() ==
             FiniteElementCollection::CONTINUOUS)
         {
            // This is a continuous field so we can evaluate it on the boundary.
            fes->GetBdrElementVDofs(T.ElementNo, vdofs);
            fe = fes->GetBE(T.ElementNo);
         }
         else
         {
            // This is a discontinuous vector field which cannot be evaluated on
            // the boundary so we'll evaluate it in the neighboring element.
            FaceElementTransformations * FET =
               fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);
            MFEM_ASSERT(FET != nullptr,
                        "FaceElementTransformation must be valid for a boundary element");

            // Boundary elements and boundary faces may have different
            // orientations so adjust the integration point if necessary.
            int f, o;
            fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
            IntegrationPoint fip =
               Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o, ip);

            // Compute and set the point in element 1 from fip
            FET->SetAllIntPoints(&fip);
            ElementTransformation & T1 = FET->GetElement1Transformation();
            return GetVectorValue(T1, T1.GetIntPoint(), val);
         }
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);
         MFEM_ASSERT(FET != nullptr,
                     "FaceElementTransformation must be valid for a boundary element");

         // Evaluate in neighboring element for both continuous and
         // discontinuous fields (the integration point in T1 should have
         // already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         return GetVectorValue(T1, T1.GetIntPoint(), val);
      }
      default:
      {
         MFEM_ABORT("GridFunction::GetVectorValue: Unsupported element type \""
                    << T.ElementType << "\"");
         if (val.Size() > 0) { val = NAN; }
         return;
      }
   }

   int dof = fe->GetDof();
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }
   if (fe->GetRangeType() == FiniteElement::SCALAR)
   {
      Vector shape(dof);
      if (fe->GetMapType() == FiniteElement::VALUE)
      {
         fe->CalcShape(ip, shape);
      }
      else
      {
         fe->CalcPhysShape(T, shape);
      }
      int vdim = fes->GetVDim();
      val.SetSize(vdim);
      for (int k = 0; k < vdim; k++)
      {
         val(k) = shape * (&loc_data[dof * k]);
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      int vdim = std::max(spaceDim, fe->GetRangeDim());
      DenseMatrix vshape(dof, vdim);
      fe->CalcVShape(T, vshape);
      val.SetSize(vdim);
      vshape.MultTranspose(loc_data, val);
   }
}

void GridFunction::GetVectorValues(ElementTransformation &T,
                                   const IntegrationRule &ir,
                                   DenseMatrix &vals,
                                   DenseMatrix *tr) const
{
   if (tr)
   {
      T.Transform(ir, *tr);
   }

   const FiniteElement *FElem = fes->GetFE(T.ElementNo);
   int dof = FElem->GetDof();

   Array<int> vdofs;
   DofTransformation * doftrans = fes->GetElementVDofs(T.ElementNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }

   int nip = ir.GetNPoints();

   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      Vector shape(dof);
      int vdim = fes->GetVDim();
      vals.SetSize(vdim, nip);
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T.SetIntPoint(&ip);
         FElem->CalcPhysShape(T, shape);

         for (int k = 0; k < vdim; k++)
         {
            vals(k,j) = shape * (&loc_data[dof * k]);
         }
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      int vdim = std::max(spaceDim, FElem->GetRangeDim());
      DenseMatrix vshape(dof, vdim);

      vals.SetSize(vdim, nip);
      Vector val_j;

      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T.SetIntPoint(&ip);
         FElem->CalcVShape(T, vshape);

         vals.GetColumnReference(j, val_j);
         vshape.MultTranspose(loc_data, val_j);
      }
   }
}

int GridFunction::GetFaceVectorValues(
   int i, int side, const IntegrationRule &ir,
   DenseMatrix &vals, DenseMatrix &tr) const
{
   int di;
   FaceElementTransformations *Transf;

   IntegrationRule eir(ir.GetNPoints());  // ---
   Transf = fes->GetMesh()->GetFaceElementTransformations(i, 0);
   if (side == 2)
   {
      if (Transf->Elem2No < 0 ||
          fes->GetAttribute(Transf->Elem1No) <=
          fes->GetAttribute(Transf->Elem2No))
      {
         di = 0;
      }
      else
      {
         di = 1;
      }
   }
   else
   {
      di = side;
   }
   if (di == 0)
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 5);
      MFEM_ASSERT(Transf != nullptr, "FaceElementTransformation cannot be null!");
      Transf->Loc1.Transform(ir, eir);
      GetVectorValues(*Transf->Elem1, eir, vals, &tr);
   }
   else
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 10);
      MFEM_ASSERT(Transf != nullptr, "FaceElementTransformation cannot be null!");
      Transf->Loc2.Transform(ir, eir);
      GetVectorValues(*Transf->Elem2, eir, vals, &tr);
   }

   return di;
}

void GridFunction::GetValuesFrom(const GridFunction &orig_func)
{
   // Without averaging ...

   const FiniteElementSpace *orig_fes = orig_func.FESpace();
   DofTransformation * doftrans;
   DofTransformation * orig_doftrans;
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, orig_loc_values;
   int i, j, d, ne, dof, odof, vdim;

   ne = fes->GetNE();
   vdim = fes->GetVDim();
   for (i = 0; i < ne; i++)
   {
      doftrans = fes->GetElementVDofs(i, vdofs);
      orig_doftrans = orig_fes->GetElementVDofs(i, orig_vdofs);
      orig_func.GetSubVector(orig_vdofs, orig_loc_values);
      if (orig_doftrans)
      {
         orig_doftrans->InvTransformPrimal(orig_loc_values);
      }
      const FiniteElement *fe = fes->GetFE(i);
      const FiniteElement *orig_fe = orig_fes->GetFE(i);
      dof = fe->GetDof();
      odof = orig_fe->GetDof();
      loc_values.SetSize(dof * vdim);
      shape.SetSize(odof);
      const IntegrationRule &ir = fe->GetNodes();
      for (j = 0; j < dof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         orig_fe->CalcShape(ip, shape);
         for (d = 0; d < vdim; d++)
         {
            loc_values(d*dof+j) = shape * (&orig_loc_values[d * odof]);
         }
      }
      if (doftrans)
      {
         doftrans->TransformPrimal(loc_values);
      }
      SetSubVector(vdofs, loc_values);
   }
}

void GridFunction::GetBdrValuesFrom(const GridFunction &orig_func)
{
   // Without averaging ...

   const FiniteElementSpace *orig_fes = orig_func.FESpace();
   // DofTransformation * doftrans;
   // DofTransformation * orig_doftrans;
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, loc_values_t, orig_loc_values, orig_loc_values_t;
   int i, j, d, nbe, dof, odof, vdim;

   nbe = fes->GetNBE();
   vdim = fes->GetVDim();
   for (i = 0; i < nbe; i++)
   {
      fes->GetBdrElementVDofs(i, vdofs);
      orig_fes->GetBdrElementVDofs(i, orig_vdofs);
      orig_func.GetSubVector(orig_vdofs, orig_loc_values);
      const FiniteElement *fe = fes->GetBE(i);
      const FiniteElement *orig_fe = orig_fes->GetBE(i);
      dof = fe->GetDof();
      odof = orig_fe->GetDof();
      loc_values.SetSize(dof * vdim);
      shape.SetSize(odof);
      const IntegrationRule &ir = fe->GetNodes();
      for (j = 0; j < dof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         orig_fe->CalcShape(ip, shape);
         for (d = 0; d < vdim; d++)
         {
            loc_values(d*dof+j) = shape * (&orig_loc_values[d * odof]);
         }
      }
      SetSubVector(vdofs, loc_values);
   }
}

void GridFunction::GetVectorFieldValues(
   int i, const IntegrationRule &ir, DenseMatrix &vals,
   DenseMatrix &tr, int comp) const
{
   Array<int> vdofs;
   ElementTransformation *transf;

   int d, k, n, sdim, dof;

   n = ir.GetNPoints();
   DofTransformation * doftrans = fes->GetElementVDofs(i, vdofs);
   const FiniteElement *fe = fes->GetFE(i);
   dof = fe->GetDof();
   sdim = fes->GetMesh()->SpaceDimension();
   // int *dofs = &vdofs[comp*dof];
   transf = fes->GetElementTransformation(i);
   transf->Transform(ir, tr);
   vals.SetSize(n, sdim);
   DenseMatrix vshape(dof, sdim);
   Vector loc_data, val(sdim);
   GetSubVector(vdofs, loc_data);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(loc_data);
   }
   for (k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      transf->SetIntPoint(&ip);
      fe->CalcVShape(*transf, vshape);
      vshape.MultTranspose(loc_data, val);
      for (d = 0; d < sdim; d++)
      {
         vals(k,d) = val(d);
      }
   }
}

void GridFunction::ReorderByNodes()
{
   if (fes->GetOrdering() == Ordering::byNODES)
   {
      return;
   }

   int i, j, k;
   int vdim = fes->GetVDim();
   int ndofs = fes->GetNDofs();
   real_t *temp = new real_t[size];

   k = 0;
   for (j = 0; j < ndofs; j++)
      for (i = 0; i < vdim; i++)
      {
         temp[j+i*ndofs] = data[k++];
      }

   for (i = 0; i < size; i++)
   {
      data[i] = temp[i];
   }

   delete [] temp;
}

void GridFunction::GetVectorFieldNodalValues(Vector &val, int comp) const
{
   int i, k;
   Array<int> overlap(fes->GetNV());
   Array<int> vertices;
   DenseMatrix vals, tr;

   val.SetSize(overlap.Size());
   overlap = 0;
   val = 0.0;

   comp--;
   for (i = 0; i < fes->GetNE(); i++)
   {
      const IntegrationRule *ir =
         Geometries.GetVertices(fes->GetFE(i)->GetGeomType());
      fes->GetElementVertices(i, vertices);
      GetVectorFieldValues(i, *ir, vals, tr);
      for (k = 0; k < ir->GetNPoints(); k++)
      {
         val(vertices[k]) += vals(k, comp);
         overlap[vertices[k]]++;
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      val(i) /= overlap[i];
   }
}

void GridFunction::ProjectVectorFieldOn(GridFunction &vec_field, int comp)
{
   FiniteElementSpace *new_fes = vec_field.FESpace();

   int d, i, k, ind, dof, sdim;
   Array<int> overlap(new_fes->GetVSize());
   Array<int> new_vdofs;
   DenseMatrix vals, tr;

   sdim = fes->GetMesh()->SpaceDimension();
   overlap = 0;
   vec_field = 0.0;

   for (i = 0; i < new_fes->GetNE(); i++)
   {
      const FiniteElement *fe = new_fes->GetFE(i);
      const IntegrationRule &ir = fe->GetNodes();
      GetVectorFieldValues(i, ir, vals, tr, comp);
      new_fes->GetElementVDofs(i, new_vdofs);
      dof = fe->GetDof();
      for (d = 0; d < sdim; d++)
      {
         for (k = 0; k < dof; k++)
         {
            if ( (ind=new_vdofs[dof*d+k]) < 0 )
            {
               ind = -1-ind, vals(k, d) = - vals(k, d);
            }
            vec_field(ind) += vals(k, d);
            overlap[ind]++;
         }
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      vec_field(i) /= overlap[i];
   }
}

void GridFunction::AccumulateAndCountDerivativeValues(
   int comp, int der_comp, GridFunction &der,
   Array<int> &zones_per_dof) const
{
   FiniteElementSpace * der_fes = der.FESpace();
   ElementTransformation * transf;
   zones_per_dof.SetSize(der_fes->GetVSize());
   Array<int> der_dofs, vdofs;
   DenseMatrix dshape, inv_jac;
   Vector pt_grad, loc_func;
   int i, j, k, dim, dof, der_dof, ind;
   real_t a;

   zones_per_dof = 0;
   der = 0.0;

   comp--;
   for (i = 0; i < der_fes->GetNE(); i++)
   {
      const FiniteElement *der_fe = der_fes->GetFE(i);
      const FiniteElement *fe = fes->GetFE(i);
      const IntegrationRule &ir = der_fe->GetNodes();
      der_fes->GetElementDofs(i, der_dofs);
      fes->GetElementVDofs(i, vdofs);
      dim = fe->GetDim();
      dof = fe->GetDof();
      der_dof = der_fe->GetDof();
      dshape.SetSize(dof, dim);
      inv_jac.SetSize(dim);
      pt_grad.SetSize(dim);
      loc_func.SetSize(dof);
      transf = fes->GetElementTransformation(i);
      for (j = 0; j < dof; j++)
         loc_func(j) = ( (ind=vdofs[comp*dof+j]) >= 0 ) ?
                       (data[ind]) : (-data[-1-ind]);
      for (k = 0; k < der_dof; k++)
      {
         const IntegrationPoint &ip = ir.IntPoint(k);
         fe->CalcDShape(ip, dshape);
         dshape.MultTranspose(loc_func, pt_grad);
         transf->SetIntPoint(&ip);
         CalcInverse(transf->Jacobian(), inv_jac);
         a = 0.0;
         for (j = 0; j < dim; j++)
         {
            a += inv_jac(j, der_comp) * pt_grad(j);
         }
         der(der_dofs[k]) += a;
         zones_per_dof[der_dofs[k]]++;
      }
   }
}

void GridFunction::GetDerivative(int comp, int der_comp,
                                 GridFunction &der) const
{
   Array<int> overlap;
   AccumulateAndCountDerivativeValues(comp, der_comp, der, overlap);

   for (int i = 0; i < overlap.Size(); i++)
   {
      der(i) /= overlap[i];
   }
}

void GridFunction::GetVectorGradientHat(
   ElementTransformation &T, DenseMatrix &gh) const
{
   const FiniteElement *FElem = fes->GetFE(T.ElementNo);
   int dim = FElem->GetDim(), dof = FElem->GetDof();
   Vector loc_data;
   GetElementDofValues(T.ElementNo, loc_data);
   // assuming scalar FE
   int vdim = fes->GetVDim();
   DenseMatrix dshape(dof, dim);
   FElem->CalcDShape(T.GetIntPoint(), dshape);
   gh.SetSize(vdim, dim);
   DenseMatrix loc_data_mat(loc_data.GetData(), dof, vdim);
   MultAtB(loc_data_mat, dshape, gh);
}

real_t GridFunction::GetDivergence(ElementTransformation &T) const
{
   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
      {
         int elNo = T.ElementNo;
         const FiniteElement *fe = fes->GetFE(elNo);
         if (fe->GetRangeType() == FiniteElement::SCALAR)
         {
            MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE,
                        "invalid FE map type");
            DenseMatrix grad_hat;
            GetVectorGradientHat(T, grad_hat);
            const DenseMatrix &Jinv = T.InverseJacobian();
            real_t div_v = 0.0;
            for (int i = 0; i < Jinv.Width(); i++)
            {
               for (int j = 0; j < Jinv.Height(); j++)
               {
                  div_v += grad_hat(i, j) * Jinv(j, i);
               }
            }
            return div_v;
         }
         else
         {
            // Assuming RT-type space
            Array<int> dofs;
            DofTransformation * doftrans = fes->GetElementDofs(elNo, dofs);
            Vector loc_data, divshape(fe->GetDof());
            GetSubVector(dofs, loc_data);
            if (doftrans)
            {
               doftrans->InvTransformPrimal(loc_data);
            }
            fe->CalcDivShape(T.GetIntPoint(), divshape);
            return (loc_data * divshape) / T.Weight();
         }
      }
      break;
      case ElementTransformation::BDR_ELEMENT:
      {
         // In order to properly capture the derivative of the normal component
         // of the field (as well as the transverse divergence of the
         // tangential components) we must evaluate it in the neighboring
         // element.
         FaceElementTransformations * FET =
            fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);

         // Boundary elements and boundary faces may have different
         // orientations so adjust the integration point if necessary.
         int f, o;
         fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
         IntegrationPoint fip =
            Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o,
                                            T.GetIntPoint());

         // Compute and set the point in element 1 from fip
         FET->SetAllIntPoints(&fip);
         ElementTransformation & T1 = FET->GetElement1Transformation();

         return GetDivergence(T1);
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         // This must be a DG context so this dynamic cast must succeed.
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);

         // Evaluate in neighboring element (the integration point in T1 should
         // have already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         return GetDivergence(T1);
      }
      break;
      default:
      {
         MFEM_ABORT("GridFunction::GetDivergence: Unsupported element type \""
                    << T.ElementType << "\"");
      }
   }
   return 0.0; // never reached
}

void GridFunction::GetCurl(ElementTransformation &T, Vector &curl) const
{
   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
      {
         int elNo = T.ElementNo;
         const FiniteElement *fe = fes->GetFE(elNo);
         if (fe->GetRangeType() == FiniteElement::SCALAR)
         {
            MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE,
                        "invalid FE map type");
            DenseMatrix grad_hat;
            GetVectorGradientHat(T, grad_hat);
            const DenseMatrix &Jinv = T.InverseJacobian();
            // Dimensions of grad are vdim x FElem->Dim
            DenseMatrix grad(grad_hat.Height(), Jinv.Width());
            Mult(grad_hat, Jinv, grad);
            MFEM_ASSERT(grad.Height() == grad.Width(), "");
            if (grad.Height() == 3)
            {
               curl.SetSize(3);
               curl(0) = grad(2,1) - grad(1,2);
               curl(1) = grad(0,2) - grad(2,0);
               curl(2) = grad(1,0) - grad(0,1);
            }
            else if (grad.Height() == 2)
            {
               curl.SetSize(1);
               curl(0) = grad(1,0) - grad(0,1);
            }
         }
         else
         {
            // Assuming ND-type space
            Array<int> dofs;
            DofTransformation * doftrans = fes->GetElementDofs(elNo, dofs);
            Vector loc_data;
            GetSubVector(dofs, loc_data);
            if (doftrans)
            {
               doftrans->InvTransformPrimal(loc_data);
            }
            DenseMatrix curl_shape(fe->GetDof(), fe->GetCurlDim());
            curl.SetSize(curl_shape.Width());
            fe->CalcPhysCurlShape(T, curl_shape);
            curl_shape.MultTranspose(loc_data, curl);
         }
      }
      break;
      case ElementTransformation::BDR_ELEMENT:
      {
         // In order to capture the tangential components of the curl we
         // must evaluate it in the neighboring element.
         FaceElementTransformations * FET =
            fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);

         // Boundary elements and boundary faces may have different
         // orientations so adjust the integration point if necessary.
         int f, o;
         fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
         IntegrationPoint fip =
            Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o,
                                            T.GetIntPoint());

         // Compute and set the point in element 1 from fip
         FET->SetAllIntPoints(&fip);
         ElementTransformation & T1 = FET->GetElement1Transformation();

         GetCurl(T1, curl);
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         // This must be a DG context so this dynamic cast must succeed.
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);

         // Evaluate in neighboring element (the integration point in T1 should
         // have already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         GetCurl(T1, curl);
      }
      break;
      default:
      {
         MFEM_ABORT("GridFunction::GetCurl: Unsupported element type \""
                    << T.ElementType << "\"");
      }
   }
}

void GridFunction::GetGradient(ElementTransformation &T, Vector &grad) const
{
   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
      {
         const FiniteElement *fe = fes->GetFE(T.ElementNo);
         MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE,
                     "invalid FE map type");
         MFEM_ASSERT(fes->GetVDim() == 1, "Defined for scalar functions.");
         int spaceDim = fes->GetMesh()->SpaceDimension();
         int dim = fe->GetDim(), dof = fe->GetDof();
         DenseMatrix dshape(dof, dim);
         Vector lval, gh(dim);

         grad.SetSize(spaceDim);
         GetElementDofValues(T.ElementNo, lval);
         fe->CalcDShape(T.GetIntPoint(), dshape);
         dshape.MultTranspose(lval, gh);
         T.InverseJacobian().MultTranspose(gh, grad);
      }
      break;
      case ElementTransformation::BDR_ELEMENT:
      {
         // In order to properly capture the normal component of the gradient
         // as well as its tangential components we must evaluate it in the
         // neighboring element.
         FaceElementTransformations * FET =
            fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);

         // Boundary elements and boundary faces may have different
         // orientations so adjust the integration point if necessary.
         int f, o;
         fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
         IntegrationPoint fip =
            Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o,
                                            T.GetIntPoint());

         // Compute and set the point in element 1 from fip
         FET->SetAllIntPoints(&fip);
         ElementTransformation & T1 = FET->GetElement1Transformation();

         GetGradient(T1, grad);
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         // This must be a DG context so this dynamic cast must succeed.
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);

         // Evaluate in neighboring element (the integration point in T1 should
         // have already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         GetGradient(T1, grad);
      }
      break;
      default:
      {
         MFEM_ABORT("GridFunction::GetGradient: Unsupported element type \""
                    << T.ElementType << "\"");
      }
   }
}

void GridFunction::GetGradients(ElementTransformation &tr,
                                const IntegrationRule &ir,
                                DenseMatrix &grad) const
{
   int elNo = tr.ElementNo;
   const FiniteElement *fe = fes->GetFE(elNo);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   DenseMatrix dshape(fe->GetDof(), fe->GetDim());
   Vector lval, gh(fe->GetDim()), gcol;

   GetElementDofValues(tr.ElementNo, lval);
   grad.SetSize(fe->GetDim(), ir.GetNPoints());
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      fe->CalcDShape(ip, dshape);
      dshape.MultTranspose(lval, gh);
      tr.SetIntPoint(&ip);
      grad.GetColumnReference(i, gcol);
      const DenseMatrix &Jinv = tr.InverseJacobian();
      Jinv.MultTranspose(gh, gcol);
   }
}

void GridFunction::GetVectorGradient(
   ElementTransformation &T, DenseMatrix &grad) const
{
   switch (T.ElementType)
   {
      case ElementTransformation::ELEMENT:
      {
         MFEM_ASSERT(fes->GetFE(T.ElementNo)->GetMapType() ==
                     FiniteElement::VALUE, "invalid FE map type");
         DenseMatrix grad_hat;
         GetVectorGradientHat(T, grad_hat);
         const DenseMatrix &Jinv = T.InverseJacobian();
         grad.SetSize(grad_hat.Height(), Jinv.Width());
         Mult(grad_hat, Jinv, grad);
      }
      break;
      case ElementTransformation::BDR_ELEMENT:
      {
         // In order to capture the normal component of the gradient we
         // must evaluate it in the neighboring element.
         FaceElementTransformations * FET =
            fes->GetMesh()->GetBdrFaceTransformations(T.ElementNo);

         // Boundary elements and boundary faces may have different
         // orientations so adjust the integration point if necessary.
         int f, o;
         fes->GetMesh()->GetBdrElementFace(T.ElementNo, &f, &o);
         IntegrationPoint fip =
            Mesh::TransformBdrElementToFace(FET->GetGeometryType(), o,
                                            T.GetIntPoint());

         // Compute and set the point in element 1 from fip
         FET->SetAllIntPoints(&fip);
         ElementTransformation & T1 = FET->GetElement1Transformation();

         GetVectorGradient(T1, grad);
      }
      break;
      case ElementTransformation::BDR_FACE:
      {
         // This must be a DG context so this dynamic cast must succeed.
         FaceElementTransformations * FET =
            dynamic_cast<FaceElementTransformations *>(&T);

         // Evaluate in neighboring element (the integration point in T1 should
         // have already been set).
         ElementTransformation & T1 = FET->GetElement1Transformation();
         GetVectorGradient(T1, grad);
      }
      break;
      default:
      {
         MFEM_ABORT("GridFunction::GetVectorGradient: "
                    "Unsupported element type \"" << T.ElementType << "\"");
      }
   }
}

void GridFunction::GetElementAverages(GridFunction &avgs) const
{
   MassIntegrator Mi;
   DenseMatrix loc_mass;
   DofTransformation * te_doftrans;
   DofTransformation * tr_doftrans;
   Array<int> te_dofs, tr_dofs;
   Vector loc_avgs, loc_this;
   Vector int_psi(avgs.Size());

   avgs = 0.0;
   int_psi = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      Mi.AssembleElementMatrix2(*fes->GetFE(i), *avgs.FESpace()->GetFE(i),
                                *fes->GetElementTransformation(i), loc_mass);
      tr_doftrans = fes->GetElementDofs(i, tr_dofs);
      te_doftrans = avgs.FESpace()->GetElementDofs(i, te_dofs);
      GetSubVector(tr_dofs, loc_this);
      if (tr_doftrans)
      {
         tr_doftrans->InvTransformPrimal(loc_this);
      }
      loc_avgs.SetSize(te_dofs.Size());
      loc_mass.Mult(loc_this, loc_avgs);
      if (te_doftrans)
      {
         te_doftrans->TransformPrimal(loc_avgs);
      }
      avgs.AddElementVector(te_dofs, loc_avgs);
      loc_this = 1.0; // assume the local basis for 'this' sums to 1
      loc_mass.Mult(loc_this, loc_avgs);
      int_psi.AddElementVector(te_dofs, loc_avgs);
   }
   for (int i = 0; i < avgs.Size(); i++)
   {
      avgs(i) /= int_psi(i);
   }
}

void GridFunction::GetElementDofValues(int el, Vector &dof_vals) const
{
   Array<int> dof_idx;
   DofTransformation * doftrans = fes->GetElementVDofs(el, dof_idx);
   GetSubVector(dof_idx, dof_vals);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(dof_vals);
   }
}

void GridFunction::ProjectGridFunction(const GridFunction &src)
{
   Mesh *mesh = fes->GetMesh();
   bool sameP = false;
   DenseMatrix P;

   if (!mesh->GetNE()) { return; }

   Geometry::Type geom, cached_geom = Geometry::INVALID;
   if (mesh->GetNumGeometries(mesh->Dimension()) == 1)
   {
      // Assuming that the projection matrix is the same for all elements
      sameP = true;
      fes->GetFE(0)->Project(*src.fes->GetFE(0),
                             *mesh->GetElementTransformation(0), P);
   }
   const int vdim = fes->GetVDim();
   MFEM_VERIFY(vdim == src.fes->GetVDim(), "incompatible vector dimensions!");

   Array<int> src_vdofs, dest_vdofs;
   Vector src_lvec, dest_lvec(vdim*P.Height());

   for (int i = 0; i < mesh->GetNE(); i++)
   {
      // Assuming the projection matrix P depends only on the element geometry
      if ( !sameP && (geom = mesh->GetElementBaseGeometry(i)) != cached_geom )
      {
         fes->GetFE(i)->Project(*src.fes->GetFE(i),
                                *mesh->GetElementTransformation(i), P);
         dest_lvec.SetSize(vdim*P.Height());
         cached_geom = geom;
      }

      DofTransformation * src_doftrans = src.fes->GetElementVDofs(i, src_vdofs);
      src.GetSubVector(src_vdofs, src_lvec);
      if (src_doftrans)
      {
         src_doftrans->InvTransformPrimal(src_lvec);
      }
      for (int vd = 0; vd < vdim; vd++)
      {
         P.Mult(&src_lvec[vd*P.Width()], &dest_lvec[vd*P.Height()]);
      }
      DofTransformation * doftrans = fes->GetElementVDofs(i, dest_vdofs);
      if (doftrans)
      {
         doftrans->TransformPrimal(dest_lvec);
      }
      SetSubVector(dest_vdofs, dest_lvec);
   }
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                const Vector &lo_, const Vector &hi_)
{
   Array<int> vdofs;
   DofTransformation * doftrans = fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);

   GetSubVector(vdofs, vals);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(vals);
   }

   MFEM_ASSERT(weights.Size() == size, "Different # of weights and dofs.");
   MFEM_ASSERT(lo_.Size() == size, "Different # of lower bounds and dofs.");
   MFEM_ASSERT(hi_.Size() == size, "Different # of upper bounds and dofs.");

   int max_iter = 30;
   real_t tol = 1.e-12;
   SLBQPOptimizer slbqp;
   slbqp.SetMaxIter(max_iter);
   slbqp.SetAbsTol(1.0e-18);
   slbqp.SetRelTol(tol);
   slbqp.SetBounds(lo_, hi_);
   slbqp.SetLinearConstraint(weights, weights * vals);
   slbqp.SetPrintLevel(0); // print messages only if not converged
   slbqp.Mult(vals, new_vals);

   if (doftrans)
   {
      doftrans->TransformPrimal(new_vals);
   }
   SetSubVector(vdofs, new_vals);
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                real_t min_, real_t max_)
{
   Array<int> vdofs;
   DofTransformation * doftrans = fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);
   GetSubVector(vdofs, vals);
   if (doftrans)
   {
      doftrans->InvTransformPrimal(vals);
   }

   real_t max_val = vals.Max();
   real_t min_val = vals.Min();

   if (max_val <= min_)
   {
      new_vals = min_;
      if (doftrans)
      {
         doftrans->TransformPrimal(new_vals);
      }
      SetSubVector(vdofs, new_vals);
      return;
   }

   if (min_ <= min_val && max_val <= max_)
   {
      return;
   }

   Vector minv(size), maxv(size);
   minv = (min_ > min_val) ? min_ : min_val;
   maxv = (max_ < max_val) ? max_ : max_val;

   ImposeBounds(i, weights, minv, maxv);
}

void GridFunction::RestrictConforming()
{
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   const Operator *P = fes->GetProlongationMatrix();

   if (P && R)
   {
      Vector tmp(R->Height());
      R->Mult(*this, tmp);
      P->Mult(tmp, *this);
   }
}

void GridFunction::GetNodalValues(Vector &nval, int vdim) const
{
   int i, j;
   Array<int> vertices;
   Array<real_t> values;
   Array<int> overlap(fes->GetNV());
   nval.SetSize(fes->GetNV());
   nval = 0.0;
   overlap = 0;
   nval.HostReadWrite();
   for (i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVertices(i, vertices);
      GetNodalValues(i, values, vdim);
      for (j = 0; j < vertices.Size(); j++)
      {
         nval(vertices[j]) += values[j];
         overlap[vertices[j]]++;
      }
   }
   for (i = 0; i < overlap.Size(); i++)
   {
      nval(i) /= overlap[i];
   }
}


void GridFunction::CountElementsPerVDof(Array<int> &elem_per_vdof) const
{
   elem_per_vdof.SetSize(fes->GetVSize());
   elem_per_vdof = 0;
   Array<int> vdofs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         elem_per_vdof[vdofs[j]]++;
      }
   }
}

void GridFunction::AccumulateAndCountZones(Coefficient &coeff,
                                           AvgType type,
                                           Array<int> &zones_per_vdof)
{
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   // Local interpolation
   Array<int> vdofs;
   Vector vals;
   *this = 0.0;

   HostReadWrite();

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      // Local interpolation of coeff.
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (type == HARMONIC)
         {
            MFEM_VERIFY(vals[j] != 0.0,
                        "Coefficient has zeros, harmonic avg is undefined!");
            (*this)(vdofs[j]) += 1.0 / vals[j];
         }
         else if (type == ARITHMETIC)
         {
            (*this)(vdofs[j]) += vals[j];
         }
         else { MFEM_ABORT("Not implemented"); }

         zones_per_vdof[vdofs[j]]++;
      }
   }
}

void GridFunction::AccumulateAndCountZones(VectorCoefficient &vcoeff,
                                           AvgType type,
                                           Array<int> &zones_per_vdof)
{
   zones_per_vdof.SetSize(fes->GetVSize());
   zones_per_vdof = 0;

   // Local interpolation
   Array<int> vdofs;
   Vector vals;
   *this = 0.0;

   HostReadWrite();

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      // Local interpolation of coeff.
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(vcoeff, *fes->GetElementTransformation(i), vals);

      // Accumulate values in all dofs, count the zones.
      for (int j = 0; j < vdofs.Size(); j++)
      {
         int ldof;
         int isign;
         if (vdofs[j] < 0 )
         {
            ldof = -1-vdofs[j];
            isign = -1;
         }
         else
         {
            ldof = vdofs[j];
            isign = 1;
         }

         if (type == HARMONIC)
         {
            MFEM_VERIFY(vals[j] != 0.0,
                        "Coefficient has zeros, harmonic avg is undefined!");
            (*this)(ldof) += isign / vals[j];
         }
         else if (type == ARITHMETIC)
         {
            (*this)(ldof) += isign*vals[j];

         }
         else { MFEM_ABORT("Not implemented"); }

         zones_per_vdof[ldof]++;
      }
   }
}

void GridFunction::AccumulateAndCountBdrValues(
   Coefficient *coeff[], VectorCoefficient *vcoeff, const Array<int> &attr,
   Array<int> &values_counter)
{
   Array<int> vdofs;
   Vector vc;

   values_counter.SetSize(Size());
   values_counter = 0;

   const int vdim = fes->GetVDim();
   HostReadWrite();

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (attr[fes->GetBdrAttribute(i) - 1] == 0) { continue; }

      const FiniteElement *fe = fes->GetBE(i);
      const int fdof = fe->GetDof();
      ElementTransformation *transf = fes->GetBdrElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      fes->GetBdrElementVDofs(i, vdofs);

      for (int j = 0; j < fdof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         transf->SetIntPoint(&ip);
         if (vcoeff) { vcoeff->Eval(vc, *transf, ip); }
         for (int d = 0; d < vdim; d++)
         {
            if (!vcoeff && !coeff[d]) { continue; }

            real_t val = vcoeff ? vc(d) : coeff[d]->Eval(*transf, ip);
            int ind = vdofs[fdof*d+j];
            if ( ind < 0 )
            {
               val = -val, ind = -1-ind;
            }
            if (++values_counter[ind] == 1)
            {
               (*this)(ind) = val;
            }
            else
            {
               (*this)(ind) += val;
            }
         }
      }
   }

   // In the case of partially conforming space, i.e. (fes->cP != NULL), we need
   // to set the values of all dofs on which the dofs set above depend.
   // Dependency is defined from the matrix A = cP.cR: dof i depends on dof j
   // iff A_ij != 0. It is sufficient to resolve just the first level of
   // dependency, since A is a projection matrix: A^n = A due to cR.cP = I.
   // Cases like these arise in 3D when boundary edges are constrained by
   // (depend on) internal faces/elements, or for internal boundaries in 2 or
   // 3D. We use the virtual method GetBoundaryClosure from NCMesh to resolve
   // the dependencies.
   if (fes->Nonconforming() && (fes->GetMesh()->Dimension() == 2 ||
                                fes->GetMesh()->Dimension() == 3))
   {
      Vector vals;
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices, bdr_faces;
      ncmesh->GetBoundaryClosure(attr, bdr_vertices, bdr_edges, bdr_faces);

      auto mark_dofs = [&](ElementTransformation &transf, const FiniteElement &fe)
      {
         if (!vcoeff)
         {
            vals.SetSize(fe.GetDof());
            for (int d = 0; d < vdim; d++)
            {
               if (!coeff[d]) { continue; }

               fe.Project(*coeff[d], transf, vals);
               for (int k = 0; k < vals.Size(); k++)
               {
                  const int ind = vdofs[d*vals.Size()+k];
                  if (++values_counter[ind] == 1)
                  {
                     (*this)(ind) = vals(k);
                  }
                  else
                  {
                     (*this)(ind) += vals(k);
                  }
               }
            }
         }
         else // vcoeff != NULL
         {
            vals.SetSize(vdim*fe.GetDof());
            fe.Project(*vcoeff, transf, vals);
            for (int k = 0; k < vals.Size(); k++)
            {
               const int ind = vdofs[k];
               if (++values_counter[ind] == 1)
               {
                  (*this)(ind) = vals(k);
               }
               else
               {
                  (*this)(ind) += vals(k);
               }
            }
         }
      };

      for (auto edge : bdr_edges)
      {
         fes->GetEdgeVDofs(edge, vdofs);
         if (vdofs.Size() == 0) { continue; }

         ElementTransformation *transf = mesh->GetEdgeTransformation(edge);
         const FiniteElement *fe = fes->GetEdgeElement(edge);
         mark_dofs(*transf, *fe);
      }

      for (auto face : bdr_faces)
      {
         fes->GetFaceVDofs(face, vdofs);
         if (vdofs.Size() == 0) { continue; }

         ElementTransformation *transf = mesh->GetFaceTransformation(face);
         const FiniteElement *fe = fes->GetFaceElement(face);
         mark_dofs(*transf, *fe);
      }
   }
}

static void accumulate_dofs(const Array<int> &dofs, const Vector &vals,
                            Vector &gf, Array<int> &values_counter)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      int k = dofs[i];
      real_t val = vals(i);
      if (k < 0) { k = -1 - k; val = -val; }
      if (++values_counter[k] == 1)
      {
         gf(k) = val;
      }
      else
      {
         gf(k) += val;
      }
   }
}

void GridFunction::AccumulateAndCountBdrTangentValues(
   VectorCoefficient &vcoeff, const Array<int> &bdr_attr,
   Array<int> &values_counter)
{
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   Vector lvec;

   values_counter.SetSize(Size());
   values_counter = 0;

   HostReadWrite();

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      DofTransformation *dof_tr = fes->GetBdrElementDofs(i, dofs);
      lvec.SetSize(fe->GetDof());
      fe->Project(vcoeff, *T, lvec);
      if (dof_tr) { dof_tr->TransformPrimal(lvec); }
      accumulate_dofs(dofs, lvec, *this, values_counter);
   }

   if (fes->Nonconforming() && (fes->GetMesh()->Dimension() == 2 ||
                                fes->GetMesh()->Dimension() == 3))
   {
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices, bdr_faces;
      ncmesh->GetBoundaryClosure(bdr_attr, bdr_vertices, bdr_edges, bdr_faces);

      for (auto edge : bdr_edges)
      {
         fes->GetEdgeDofs(edge, dofs);
         if (dofs.Size() == 0) { continue; }

         T = mesh->GetEdgeTransformation(edge);
         fe = fes->GetEdgeElement(edge);
         lvec.SetSize(fe->GetDof());
         fe->Project(vcoeff, *T, lvec);
         accumulate_dofs(dofs, lvec, *this, values_counter);
      }

      for (auto face : bdr_faces)
      {
         fes->GetFaceDofs(face, dofs);
         if (dofs.Size() == 0) { continue; }

         T = mesh->GetFaceTransformation(face);
         fe = fes->GetFaceElement(face);
         lvec.SetSize(fe->GetDof());
         fe->Project(vcoeff, *T, lvec);
         accumulate_dofs(dofs, lvec, *this, values_counter);
      }
   }
}

void GridFunction::ComputeMeans(AvgType type, Array<int> &zones_per_vdof)
{
   switch (type)
   {
      case ARITHMETIC:
         for (int i = 0; i < size; i++)
         {
            const int nz = zones_per_vdof[i];
            if (nz) { (*this)(i) /= nz; }
         }
         break;

      case HARMONIC:
         for (int i = 0; i < size; i++)
         {
            const int nz = zones_per_vdof[i];
            if (nz) { (*this)(i) = nz/(*this)(i); }
         }
         break;

      default:
         MFEM_ABORT("invalid AvgType");
   }
}

void GridFunction::ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                           real_t &integral)
{
   if (!fes->GetNE())
   {
      integral = 0.0;
      return;
   }

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const real_t *center = delta_coeff.Center();
   const real_t *vert = mesh->GetVertex(0);
   real_t min_dist, dist;
   int v_idx = 0;

   // find the vertex closest to the center of the delta function
   min_dist = Distance(center, vert, dim);
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      vert = mesh->GetVertex(i);
      dist = Distance(center, vert, dim);
      if (dist < min_dist)
      {
         min_dist = dist;
         v_idx = i;
      }
   }

   (*this) = 0.0;
   integral = 0.0;

   if (min_dist >= delta_coeff.Tol())
   {
      return;
   }

   // find the elements that have 'v_idx' as a vertex
   MassIntegrator Mi(*delta_coeff.Weight());
   DenseMatrix loc_mass;
   Array<int> vdofs, vertices;
   Vector vals, loc_mass_vals;
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      mesh->GetElementVertices(i, vertices);
      for (int j = 0; j < vertices.Size(); j++)
         if (vertices[j] == v_idx)
         {
            const FiniteElement *fe = fes->GetFE(i);
            Mi.AssembleElementMatrix(*fe, *fes->GetElementTransformation(i),
                                     loc_mass);
            vals.SetSize(fe->GetDof());
            fe->ProjectDelta(j, vals);
            const DofTransformation* const doftrans = fes->GetElementVDofs(i, vdofs);
            if (doftrans)
            {
               doftrans->TransformPrimal(vals);
            }
            SetSubVector(vdofs, vals);
            loc_mass_vals.SetSize(vals.Size());
            loc_mass.Mult(vals, loc_mass_vals);
            integral += loc_mass_vals.Sum(); // partition of unity basis
            break;
         }
   }
}

void GridFunction::ProjectCoefficient(Coefficient &coeff)
{
   DeltaCoefficient *delta_c = dynamic_cast<DeltaCoefficient *>(&coeff);
   DofTransformation * doftrans = NULL;

   if (delta_c == NULL)
   {
      if (fes->GetNURBSext() == NULL)
      {
         Array<int> vdofs;
         Vector vals;

         for (int i = 0; i < fes->GetNE(); i++)
         {
            doftrans = fes->GetElementVDofs(i, vdofs);
            vals.SetSize(vdofs.Size());
            fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
            if (doftrans)
            {
               doftrans->TransformPrimal(vals);
            }
            SetSubVector(vdofs, vals);
         }
      }
      else
      {
         // Define and assemble linear form
         LinearForm b(fes);
         b.AddDomainIntegrator(new DomainLFIntegrator(coeff));
         b.Assemble();

         // Define and assemble bilinear form
         BilinearForm a(fes);
         a.AddDomainIntegrator(new MassIntegrator());
         a.Assemble();

         // Set solver and preconditioner
         SparseMatrix A(a.SpMat());
         GSSmoother  prec(A);
         CGSolver cg;
         cg.SetOperator(A);
         cg.SetPreconditioner(prec);
         cg.SetRelTol(1e-12);
         cg.SetMaxIter(1000);
         cg.SetPrintLevel(0);

         // Solve and get solution
         *this = 0.0;
         cg.Mult(b,*this);
      }
   }
   else
   {
      real_t integral;

      ProjectDeltaCoefficient(*delta_c, integral);

      (*this) *= (delta_c->Scale() / integral);
   }
}

void GridFunction::ProjectCoefficient(
   Coefficient &coeff, Array<int> &dofs, int vd)
{
   int el = -1;
   ElementTransformation *T = NULL;
   const FiniteElement *fe = NULL;

   for (int i = 0; i < dofs.Size(); i++)
   {
      int dof = dofs[i], j = fes->GetElementForDof(dof);
      if (el != j)
      {
         el = j;
         T = fes->GetElementTransformation(el);
         fe = fes->GetFE(el);
      }
      int vdof = fes->DofToVDof(dof, vd);
      int ld = fes->GetLocalDofForDof(dof);
      const IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      (*this)(vdof) = coeff.Eval(*T, ip);
   }
}

void GridFunction::ProjectCoefficient(VectorCoefficient &vcoeff)
{
   if (fes->GetNURBSext() == NULL)
   {
      int i;
      Array<int> vdofs;
      Vector vals;

      DofTransformation * doftrans = NULL;

      for (i = 0; i < fes->GetNE(); i++)
      {
         doftrans = fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         fes->GetFE(i)->Project(vcoeff, *fes->GetElementTransformation(i), vals);
         if (doftrans)
         {
            doftrans->TransformPrimal(vals);
         }
         SetSubVector(vdofs, vals);
      }
   }
   else
   {
      // Define and assemble linear form
      LinearForm b(fes);
      b.AddDomainIntegrator(new VectorFEDomainLFIntegrator(vcoeff));
      b.Assemble();

      // Define and assemble bilinear form
      BilinearForm a(fes);
      a.AddDomainIntegrator(new VectorFEMassIntegrator());
      a.Assemble();

      // Set solver and preconditioner
      SparseMatrix A(a.SpMat());
      GSSmoother  prec(A);
      CGSolver cg;
      cg.SetOperator(A);
      cg.SetPreconditioner(prec);
      cg.SetRelTol(1e-12);
      cg.SetMaxIter(1000);
      cg.SetPrintLevel(0);

      // Solve and get solution
      *this = 0.0;
      cg.Mult(b,*this);
   }
}

void GridFunction::ProjectCoefficient(
   VectorCoefficient &vcoeff, Array<int> &dofs)
{
   int el = -1;
   ElementTransformation *T = NULL;
   const FiniteElement *fe = NULL;

   Vector val;

   for (int i = 0; i < dofs.Size(); i++)
   {
      int dof = dofs[i], j = fes->GetElementForDof(dof);
      if (el != j)
      {
         el = j;
         T = fes->GetElementTransformation(el);
         fe = fes->GetFE(el);
      }
      int ld = fes->GetLocalDofForDof(dof);
      const IntegrationPoint &ip = fe->GetNodes().IntPoint(ld);
      T->SetIntPoint(&ip);
      vcoeff.Eval(val, *T, ip);
      for (int vd = 0; vd < fes->GetVDim(); vd ++)
      {
         int vdof = fes->DofToVDof(dof, vd);
         (*this)(vdof) = val(vd);
      }
   }
}

void GridFunction::ProjectCoefficient(VectorCoefficient &vcoeff, int attribute)
{
   int i;
   Array<int> vdofs;
   Vector vals;

   DofTransformation * doftrans = NULL;

   for (i = 0; i < fes->GetNE(); i++)
   {
      if (fes->GetAttribute(i) != attribute)
      {
         continue;
      }

      doftrans = fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(vcoeff, *fes->GetElementTransformation(i), vals);
      if (doftrans)
      {
         doftrans->TransformPrimal(vals);
      }
      SetSubVector(vdofs, vals);
   }
}

void GridFunction::ProjectCoefficient(Coefficient *coeff[])
{
   int i, j, fdof, d, ind, vdim;
   real_t val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   // DofTransformation * doftrans;
   Array<int> vdofs;

   vdim = fes->GetVDim();
   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      // doftrans = fes->GetElementVDofs(i, vdofs);
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < fdof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         transf->SetIntPoint(&ip);
         for (d = 0; d < vdim; d++)
         {
            if (!coeff[d]) { continue; }

            val = coeff[d]->Eval(*transf, ip);
            if ( (ind = vdofs[fdof*d+j]) < 0 )
            {
               val = -val, ind = -1-ind;
            }
            (*this)(ind) = val;
         }
      }
   }
}

void GridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff,
                                          Array<int> &dof_attr)
{
   Array<int> vdofs;
   Vector vals;

   HostWrite();
   // maximal element attribute for each dof
   dof_attr.SetSize(fes->GetVSize());
   dof_attr = -1;

   // local projection
   for (int i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);

      // the values in shared dofs are determined from the element with maximal
      // attribute
      int attr = fes->GetAttribute(i);
      for (int j = 0; j < vdofs.Size(); j++)
      {
         if (attr > dof_attr[vdofs[j]])
         {
            (*this)(vdofs[j]) = vals[j];
            dof_attr[vdofs[j]] = attr;
         }
      }
   }
}

void GridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff)
{
   Array<int> dof_attr;
   ProjectDiscCoefficient(coeff, dof_attr);
}

void GridFunction::ProjectDiscCoefficient(Coefficient &coeff, AvgType type)
{
   // Harmonic  (x1 ... xn) = [ (1/x1 + ... + 1/xn) / n ]^-1.
   // Arithmetic(x1 ... xn) = (x1 + ... + xn) / n.

   Array<int> zones_per_vdof;
   AccumulateAndCountZones(coeff, type, zones_per_vdof);

   ComputeMeans(type, zones_per_vdof);
}

void GridFunction::ProjectDiscCoefficient(VectorCoefficient &coeff,
                                          AvgType type)
{
   Array<int> zones_per_vdof;
   AccumulateAndCountZones(coeff, type, zones_per_vdof);

   ComputeMeans(type, zones_per_vdof);
}

void GridFunction::ProjectBdrCoefficient(VectorCoefficient &vcoeff,
                                         const Array<int> &attr)
{
   Array<int> values_counter;
   AccumulateAndCountBdrValues(NULL, &vcoeff, attr, values_counter);
   ComputeMeans(ARITHMETIC, values_counter);

#ifdef MFEM_DEBUG
   Array<int> ess_vdofs_marker;
   fes->GetEssentialVDofs(attr, ess_vdofs_marker);
   for (int i = 0; i < values_counter.Size(); i++)
   {
      MFEM_ASSERT(bool(values_counter[i]) == bool(ess_vdofs_marker[i]),
                  "internal error");
   }
#endif
}

void GridFunction::ProjectBdrCoefficient(Coefficient *coeff[],
                                         const Array<int> &attr)
{
   Array<int> values_counter;
   // this->HostReadWrite(); // done inside the next call
   AccumulateAndCountBdrValues(coeff, NULL, attr, values_counter);
   ComputeMeans(ARITHMETIC, values_counter);

#ifdef MFEM_DEBUG
   Array<int> ess_vdofs_marker(Size());
   ess_vdofs_marker = 0;
   Array<int> component_dof_marker;
   for (int i = 0; i < fes->GetVDim(); i++)
   {
      if (!coeff[i]) { continue; }
      fes->GetEssentialVDofs(attr, component_dof_marker,i);
      for (int j = 0; j<Size(); j++)
      {
         ess_vdofs_marker[j] = bool(ess_vdofs_marker[j]) ||
                               bool(component_dof_marker[j]);
      }
   }
   for (int i = 0; i < values_counter.Size(); i++)
   {
      MFEM_ASSERT(bool(values_counter[i]) == ess_vdofs_marker[i],
                  "internal error");
   }
#endif
}

void GridFunction::ProjectBdrCoefficientNormal(
   VectorCoefficient &vcoeff, const Array<int> &bdr_attr)
{
#if 0
   // implementation for the case when the face dofs are integrals of the
   // normal component.
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   int dim = vcoeff.GetVDim();
   Vector vc(dim), nor(dim), lvec, shape;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      int intorder = 2*fe->GetOrder(); // !!!
      const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);
      int nd = fe->GetDof();
      lvec.SetSize(nd);
      shape.SetSize(nd);
      lvec = 0.0;
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         vcoeff.Eval(vc, *T, ip);
         CalcOrtho(T->Jacobian(), nor);
         fe->CalcShape(ip, shape);
         lvec.Add(ip.weight * (vc * nor), shape);
      }
      fes->GetBdrElementDofs(i, dofs);
      SetSubVector(dofs, lvec);
   }
#else
   // implementation for the case when the face dofs are scaled point
   // values of the normal component.
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   int dim = vcoeff.GetVDim();
   Vector vc(dim), nor(dim), lvec;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      lvec.SetSize(fe->GetDof());
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         vcoeff.Eval(vc, *T, ip);
         CalcOrtho(T->Jacobian(), nor);
         lvec(j) = (vc * nor);
      }
      const DofTransformation* const doftrans = fes->GetBdrElementDofs(i, dofs);
      if (doftrans)
      {
         doftrans->TransformPrimal(lvec);
      }
      SetSubVector(dofs, lvec);
   }
#endif
}

void GridFunction::ProjectBdrCoefficientTangent(
   VectorCoefficient &vcoeff, const Array<int> &bdr_attr)
{
   Array<int> values_counter;
   AccumulateAndCountBdrTangentValues(vcoeff, bdr_attr, values_counter);
   ComputeMeans(ARITHMETIC, values_counter);
#ifdef MFEM_DEBUG
   Array<int> ess_vdofs_marker;
   fes->GetEssentialVDofs(bdr_attr, ess_vdofs_marker);
   for (int i = 0; i < values_counter.Size(); i++)
   {
      MFEM_ASSERT(bool(values_counter[i]) == bool(ess_vdofs_marker[i]),
                  "internal error");
   }
#endif
}

real_t GridFunction::ComputeL2Error(
   Coefficient *exsol[], const IntegrationRule *irs[],
   const Array<int> *elems) const
{
   real_t error = 0.0, a;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector shape;
   Array<int> vdofs;
   int fdof, d, i, intorder, j, k;

   for (i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0) { continue; }
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      shape.SetSize(fdof);
      intorder = 2*fe->GetOrder() + 3; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementVDofs(i, vdofs);
      real_t elem_error = 0.0;
      for (j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         transf->SetIntPoint(&ip);
         fe->CalcPhysShape(*transf, shape);
         for (d = 0; d < fes->GetVDim(); d++)
         {
            a = 0;
            for (k = 0; k < fdof; k++)
               if (vdofs[fdof*d+k] >= 0)
               {
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               }
               else
               {
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
               }
            a -= exsol[d]->Eval(*transf, ip);
            elem_error += ip.weight * transf->Weight() * a * a;
         }
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(elem_error);
   }

   return sqrt(error);
}

real_t GridFunction::ComputeL2Error(
   VectorCoefficient &exsol, const IntegrationRule *irs[],
   const Array<int> *elems) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0) { continue; }
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 3; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      real_t elem_error = 0.0;
      T = fes->GetElementTransformation(i);
      GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      vals.Norm2(loc_errs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         elem_error += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(elem_error);
   }
   return sqrt(error);
}

real_t GridFunction::ComputeElementGradError(int ielem,
                                             VectorCoefficient *exgrad,
                                             const IntegrationRule *irs[]) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *Tr;
   Array<int> dofs;
   Vector grad;
   int intorder;
   int dim = fes->GetMesh()->SpaceDimension();
   Vector vec(dim);

   fe = fes->GetFE(ielem);
   Tr = fes->GetElementTransformation(ielem);
   intorder = 2*fe->GetOrder() + 3; // <--------
   const IntegrationRule *ir;
   if (irs)
   {
      ir = irs[fe->GetGeomType()];
   }
   else
   {
      ir = &(IntRules.Get(fe->GetGeomType(), intorder));
   }
   fes->GetElementDofs(ielem, dofs);
   for (int j = 0; j < ir->GetNPoints(); j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      Tr->SetIntPoint(&ip);
      GetGradient(*Tr,grad);
      exgrad->Eval(vec,*Tr,ip);
      vec-=grad;
      error += ip.weight * Tr->Weight() * (vec * vec);
   }
   return sqrt(fabs(error));
}

real_t GridFunction::ComputeGradError(VectorCoefficient *exgrad,
                                      const IntegrationRule *irs[]) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *Tr;
   Array<int> dofs;
   Vector grad;
   int intorder;
   int dim = fes->GetMesh()->SpaceDimension();
   Vector vec(dim);

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      Tr = fes->GetElementTransformation(i);
      intorder = 2*fe->GetOrder() + 3; // <--------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementDofs(i, dofs);
      real_t elem_error = 0.0;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Tr->SetIntPoint(&ip);
         GetGradient(*Tr,grad);
         exgrad->Eval(vec,*Tr,ip);
         vec-=grad;
         elem_error += ip.weight * Tr->Weight() * (vec * vec);
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(elem_error);
   }
   return sqrt(error);
}

real_t GridFunction::ComputeCurlError(VectorCoefficient *excurl,
                                      const IntegrationRule *irs[]) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *Tr;
   Array<int> dofs;
   int intorder;
   int n = CurlDim();
   Vector curl(n);
   Vector vec(n);

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      Tr = fes->GetElementTransformation(i);
      intorder = 2*fe->GetOrder() + 3;
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementDofs(i, dofs);
      real_t elem_error = 0.0;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Tr->SetIntPoint(&ip);
         GetCurl(*Tr,curl);
         excurl->Eval(vec,*Tr,ip);
         vec-=curl;
         elem_error += ip.weight * Tr->Weight() * ( vec * vec );
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(elem_error);
   }

   return sqrt(error);
}

real_t GridFunction::ComputeDivError(
   Coefficient *exdiv, const IntegrationRule *irs[]) const
{
   real_t error = 0.0, a;
   const FiniteElement *fe;
   ElementTransformation *Tr;
   Array<int> dofs;
   int intorder;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      Tr = fes->GetElementTransformation(i);
      intorder = 2*fe->GetOrder() + 3;
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementDofs(i, dofs);
      real_t elem_error = 0.0;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Tr->SetIntPoint (&ip);
         a = GetDivergence(*Tr) - exdiv->Eval(*Tr, ip);
         elem_error += ip.weight * Tr->Weight() * a * a;
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(elem_error);
   }

   return sqrt(error);
}

real_t GridFunction::ComputeDGFaceJumpError(Coefficient *exsol,
                                            Coefficient *ell_coeff,
                                            class JumpScaling jump_scaling,
                                            const IntegrationRule *irs[])  const
{
   int fdof, intorder, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   FaceElementTransformations *face_elem_transf;
   Vector shape, el_dofs, err_val, ell_coeff_val;
   Array<int> vdofs;
   IntegrationPoint eip;
   real_t error = 0.0;

   mesh = fes->GetMesh();

   for (int i = 0; i < mesh->GetNumFaces(); i++)
   {
      int i1, i2;
      mesh->GetFaceElements(i, &i1, &i2);
      real_t h = mesh->GetElementSize(i1);
      intorder = fes->GetFE(i1)->GetOrder();
      if (i2 >= 0)
      {
         if ( (k = fes->GetFE(i2)->GetOrder()) > intorder )
         {
            intorder = k;
         }
         h = std::min(h, mesh->GetElementSize(i2));
      }
      int p = intorder;
      intorder = 2 * intorder;  // <-------------
      face_elem_transf = mesh->GetFaceElementTransformations(i, 5);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[face_elem_transf->GetGeometryType()];
      }
      else
      {
         ir = &(IntRules.Get(face_elem_transf->GetGeometryType(), intorder));
      }
      err_val.SetSize(ir->GetNPoints());
      ell_coeff_val.SetSize(ir->GetNPoints());
      // side 1
      transf = face_elem_transf->Elem1;
      fe = fes->GetFE(i1);
      fdof = fe->GetDof();
      fes->GetElementVDofs(i1, vdofs);
      shape.SetSize(fdof);
      el_dofs.SetSize(fdof);
      for (k = 0; k < fdof; k++)
         if (vdofs[k] >= 0)
         {
            el_dofs(k) =   (*this)(vdofs[k]);
         }
         else
         {
            el_dofs(k) = - (*this)(-1-vdofs[k]);
         }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         face_elem_transf->Loc1.Transform(ir->IntPoint(j), eip);
         fe->CalcShape(eip, shape);
         transf->SetIntPoint(&eip);
         ell_coeff_val(j) = ell_coeff->Eval(*transf, eip);
         err_val(j) = exsol->Eval(*transf, eip) - (shape * el_dofs);
      }
      if (i2 >= 0)
      {
         // side 2
         face_elem_transf = mesh->GetFaceElementTransformations(i, 10);
         transf = face_elem_transf->Elem2;
         fe = fes->GetFE(i2);
         fdof = fe->GetDof();
         fes->GetElementVDofs(i2, vdofs);
         shape.SetSize(fdof);
         el_dofs.SetSize(fdof);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) =   (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = - (*this)(-1-vdofs[k]);
            }
         for (int j = 0; j < ir->GetNPoints(); j++)
         {
            face_elem_transf->Loc2.Transform(ir->IntPoint(j), eip);
            fe->CalcShape(eip, shape);
            transf->SetIntPoint(&eip);
            ell_coeff_val(j) += ell_coeff->Eval(*transf, eip);
            ell_coeff_val(j) *= 0.5;
            err_val(j) -= (exsol->Eval(*transf, eip) - (shape * el_dofs));
         }
      }
      real_t face_error = 0.0;
      face_elem_transf = mesh->GetFaceElementTransformations(i, 16);
      transf = face_elem_transf;
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         transf->SetIntPoint(&ip);
         real_t nu = jump_scaling.Eval(h, p);
         face_error += (ip.weight * nu * ell_coeff_val(j) *
                        transf->Weight() *
                        err_val(j) * err_val(j));
      }
      // negative quadrature weights may cause the error to be negative
      error += fabs(face_error);
   }

   return sqrt(error);
}

real_t GridFunction::ComputeDGFaceJumpError(Coefficient *exsol,
                                            Coefficient *ell_coeff,
                                            real_t Nu,
                                            const IntegrationRule *irs[])  const
{
   return ComputeDGFaceJumpError(
             exsol, ell_coeff, {Nu, JumpScaling::ONE_OVER_H}, irs);
}

real_t GridFunction::ComputeH1Error(Coefficient *exsol,
                                    VectorCoefficient *exgrad,
                                    Coefficient *ell_coef, real_t Nu,
                                    int norm_type) const
{
   real_t error1 = 0.0;
   real_t error2 = 0.0;
   if (norm_type & 1) { error1 = GridFunction::ComputeGradError(exgrad); }
   if (norm_type & 2)
   {
      error2 = GridFunction::ComputeDGFaceJumpError(
                  exsol, ell_coef, {Nu, JumpScaling::ONE_OVER_H});
   }

   return sqrt(error1 * error1 + error2 * error2);
}

real_t GridFunction::ComputeH1Error(Coefficient *exsol,
                                    VectorCoefficient *exgrad,
                                    const IntegrationRule *irs[]) const
{
   real_t L2error = GridFunction::ComputeLpError(2.0,*exsol,NULL,irs);
   real_t GradError = GridFunction::ComputeGradError(exgrad,irs);
   return sqrt(L2error*L2error + GradError*GradError);
}

real_t GridFunction::ComputeHDivError(VectorCoefficient *exsol,
                                      Coefficient *exdiv,
                                      const IntegrationRule *irs[]) const
{
   real_t L2error = GridFunction::ComputeLpError(2.0,*exsol,NULL,NULL,irs);
   real_t DivError = GridFunction::ComputeDivError(exdiv,irs);
   return sqrt(L2error*L2error + DivError*DivError);
}

real_t GridFunction::ComputeHCurlError(VectorCoefficient *exsol,
                                       VectorCoefficient *excurl,
                                       const IntegrationRule *irs[]) const
{
   real_t L2error = GridFunction::ComputeLpError(2.0,*exsol,NULL,NULL,irs);
   real_t CurlError = GridFunction::ComputeCurlError(excurl,irs);
   return sqrt(L2error*L2error + CurlError*CurlError);
}

real_t GridFunction::ComputeMaxError(
   Coefficient *exsol[], const IntegrationRule *irs[]) const
{
   real_t error = 0.0, a;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector shape;
   Array<int> vdofs;
   int fdof, d, i, intorder, j, k;

   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      shape.SetSize(fdof);
      intorder = 2*fe->GetOrder() + 3; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
         transf->SetIntPoint(&ip);
         for (d = 0; d < fes->GetVDim(); d++)
         {
            a = 0;
            for (k = 0; k < fdof; k++)
               if (vdofs[fdof*d+k] >= 0)
               {
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               }
               else
               {
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
               }
            a -= exsol[d]->Eval(*transf, ip);
            a = fabs(a);
            if (error < a)
            {
               error = a;
            }
         }
      }
   }
   return error;
}

real_t GridFunction::ComputeW11Error(
   Coefficient *exsol, VectorCoefficient *exgrad, int norm_type,
   const Array<int> *elems, const IntegrationRule *irs[]) const
{
   // assuming vdim is 1
   int i, fdof, dim, intorder, j, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector e_grad, a_grad, shape, el_dofs, err_val, ell_coeff_val;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   real_t a, error = 0.0;

   mesh = fes->GetMesh();
   dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   Jinv.SetSize(dim);

   if (norm_type & 1) // L_1 norm
      for (i = 0; i < mesh->GetNE(); i++)
      {
         if (elems != NULL && (*elems)[i] == 0) { continue; }
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = fes->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         shape.SetSize(fdof);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
         {
            ir = irs[fe->GetGeomType()];
         }
         else
         {
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         real_t elem_error = 0.0;
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) = (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = -(*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            fe->CalcShape(ip, shape);
            transf->SetIntPoint(&ip);
            a = (el_dofs * shape) - (exsol->Eval(*transf, ip));
            elem_error += ip.weight * transf->Weight() * fabs(a);
         }
         error += fabs(elem_error);
      }

   if (norm_type & 2) // W^1_1 seminorm
      for (i = 0; i < mesh->GetNE(); i++)
      {
         if (elems != NULL && (*elems)[i] == 0) { continue; }
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = mesh->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         dshape.SetSize(fdof, dim);
         dshapet.SetSize(fdof, dim);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
         {
            ir = irs[fe->GetGeomType()];
         }
         else
         {
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         }
         real_t elem_error = 0.0;
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) = (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = -(*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir->GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir->IntPoint(j);
            fe->CalcDShape(ip, dshape);
            transf->SetIntPoint(&ip);
            exgrad->Eval(e_grad, *transf, ip);
            CalcInverse(transf->Jacobian(), Jinv);
            Mult(dshape, Jinv, dshapet);
            dshapet.MultTranspose(el_dofs, a_grad);
            e_grad -= a_grad;
            elem_error += ip.weight * transf->Weight() * e_grad.Norml1();
         }
         error += fabs(elem_error);
      }

   return error;
}

real_t GridFunction::ComputeLpError(const real_t p, Coefficient &exsol,
                                    Coefficient *weight,
                                    const IntegrationRule *irs[],
                                    const Array<int> *elems) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0) { continue; }
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 3; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      real_t elem_error = 0.0;
      GetValues(i, *ir, vals);
      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         real_t diff = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < infinity())
         {
            diff = pow(diff, p);
            if (weight)
            {
               diff *= weight->Eval(*T, ip);
            }
            elem_error += ip.weight * T->Weight() * diff;
         }
         else
         {
            if (weight)
            {
               diff *= weight->Eval(*T, ip);
            }
            error = std::max(error, diff);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         error += fabs(elem_error);
      }
   }

   if (p < infinity())
   {
      error = pow(error, 1./p);
   }

   return error;
}

void GridFunction::ComputeElementLpErrors(const real_t p, Coefficient &exsol,
                                          Vector &error,
                                          Coefficient *weight,
                                          const IntegrationRule *irs[]) const
{
   MFEM_ASSERT(error.Size() == fes->GetNE(),
               "Incorrect size for result vector");

   error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   Vector vals;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 3; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      GetValues(i, *ir, vals);
      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         real_t diff = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < infinity())
         {
            diff = pow(diff, p);
            if (weight)
            {
               diff *= weight->Eval(*T, ip);
            }
            error[i] += ip.weight * T->Weight() * diff;
         }
         else
         {
            if (weight)
            {
               diff *= weight->Eval(*T, ip);
            }
            error[i] = std::max(error[i], diff);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         error[i] = pow(fabs(error[i]), 1./p);
      }
   }
}

real_t GridFunction::ComputeLpError(const real_t p, VectorCoefficient &exsol,
                                    Coefficient *weight,
                                    VectorCoefficient *v_weight,
                                    const IntegrationRule *irs[]) const
{
   real_t error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 3; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      real_t elem_error = 0.0;
      T = fes->GetElementTransformation(i);
      GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      if (!v_weight)
      {
         // compute the lengths of the errors at the integration points
         // thus the vector norm is rotationally invariant
         vals.Norm2(loc_errs);
      }
      else
      {
         v_weight->Eval(exact_vals, *T, *ir);
         // column-wise dot product of the vector error (in vals) and the
         // vector weight (in exact_vals)
         for (int j = 0; j < vals.Width(); j++)
         {
            real_t errj = 0.0;
            for (int d = 0; d < vals.Height(); d++)
            {
               errj += vals(d,j)*exact_vals(d,j);
            }
            loc_errs(j) = fabs(errj);
         }
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         real_t errj = loc_errs(j);
         if (p < infinity())
         {
            errj = pow(errj, p);
            if (weight)
            {
               errj *= weight->Eval(*T, ip);
            }
            elem_error += ip.weight * T->Weight() * errj;
         }
         else
         {
            if (weight)
            {
               errj *= weight->Eval(*T, ip);
            }
            error = std::max(error, errj);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         error += fabs(elem_error);
      }
   }

   if (p < infinity())
   {
      error = pow(error, 1./p);
   }

   return error;
}

void GridFunction::ComputeElementLpErrors(const real_t p,
                                          VectorCoefficient &exsol,
                                          Vector &error,
                                          Coefficient *weight,
                                          VectorCoefficient *v_weight,
                                          const IntegrationRule *irs[]) const
{
   MFEM_ASSERT(error.Size() == fes->GetNE(),
               "Incorrect size for result vector");

   error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         int intorder = 2*fe->GetOrder() + 3; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      T = fes->GetElementTransformation(i);
      GetVectorValues(*T, *ir, vals);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      if (!v_weight)
      {
         // compute the lengths of the errors at the integration points thus the
         // vector norm is rotationally invariant
         vals.Norm2(loc_errs);
      }
      else
      {
         v_weight->Eval(exact_vals, *T, *ir);
         // column-wise dot product of the vector error (in vals) and the vector
         // weight (in exact_vals)
         for (int j = 0; j < vals.Width(); j++)
         {
            real_t errj = 0.0;
            for (int d = 0; d < vals.Height(); d++)
            {
               errj += vals(d,j)*exact_vals(d,j);
            }
            loc_errs(j) = fabs(errj);
         }
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         real_t errj = loc_errs(j);
         if (p < infinity())
         {
            errj = pow(errj, p);
            if (weight)
            {
               errj *= weight->Eval(*T, ip);
            }
            error[i] += ip.weight * T->Weight() * errj;
         }
         else
         {
            if (weight)
            {
               errj *= weight->Eval(*T, ip);
            }
            error[i] = std::max(error[i], errj);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         error[i] = pow(fabs(error[i]), 1./p);
      }
   }
}

GridFunction & GridFunction::operator=(real_t value)
{
   Vector::operator=(value);
   return *this;
}

GridFunction & GridFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetVSize(), "");
   Vector::operator=(v);
   return *this;
}

void GridFunction::Save(std::ostream &os) const
{
   fes->Save(os);
   os << '\n';
#if 0
   // Testing: write NURBS GridFunctions using "NURBS_patches" format.
   if (fes->GetNURBSext())
   {
      os << "NURBS_patches\n";
      fes->GetNURBSext()->PrintSolution(*this, os);
      os.flush();
      return;
   }
#endif
   if (fes->GetOrdering() == Ordering::byNODES)
   {
      Vector::Print(os, 1);
   }
   else
   {
      Vector::Print(os, fes->GetVDim());
   }
   os.flush();
}

void GridFunction::Save(const char *fname, int precision) const
{
   ofstream ofs(fname);
   ofs.precision(precision);
   Save(ofs);
}

#ifdef MFEM_USE_ADIOS2
void GridFunction::Save(adios2stream &os,
                        const std::string& variable_name,
                        const adios2stream::data_type type) const
{
   os.Save(*this, variable_name, type);
}
#endif

void GridFunction::SaveVTK(std::ostream &os, const std::string &field_name,
                           int ref)
{
   Mesh *mesh = fes->GetMesh();
   RefinedGeometry *RefG;
   Vector val;
   DenseMatrix vval, pmat;
   int vec_dim = VectorDim();

   if (vec_dim == 1)
   {
      // scalar data
      os << "SCALARS " << field_name << " double 1\n"
         << "LOOKUP_TABLE default\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         GetValues(i, RefG->RefPts, val, pmat);

         for (int j = 0; j < val.Size(); j++)
         {
            os << val(j) << '\n';
         }
      }
   }
   else if ( (vec_dim == 2 || vec_dim == 3) && mesh->SpaceDimension() > 1)
   {
      // vector data
      os << "VECTORS " << field_name << " double\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         // GetVectorValues(i, RefG->RefPts, vval, pmat);
         ElementTransformation * T = mesh->GetElementTransformation(i);
         GetVectorValues(*T, RefG->RefPts, vval, &pmat);

         for (int j = 0; j < vval.Width(); j++)
         {
            os << vval(0, j) << ' ' << vval(1, j) << ' ';
            if (vval.Height() == 2)
            {
               os << 0.0;
            }
            else
            {
               os << vval(2, j);
            }
            os << '\n';
         }
      }
   }
   else
   {
      // other data: save the components as separate scalars
      for (int vd = 0; vd < vec_dim; vd++)
      {
         os << "SCALARS " << field_name << vd << " double 1\n"
            << "LOOKUP_TABLE default\n";
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            RefG = GlobGeometryRefiner.Refine(
                      mesh->GetElementBaseGeometry(i), ref, 1);

            GetValues(i, RefG->RefPts, val, pmat, vd + 1);

            for (int j = 0; j < val.Size(); j++)
            {
               os << val(j) << '\n';
            }
         }
      }
   }
   os.flush();
}

void GridFunction::SaveSTLTri(std::ostream &os, real_t p1[], real_t p2[],
                              real_t p3[])
{
   real_t v1[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
   real_t v2[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };
   real_t n[] = {  v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0]
                };
   real_t rl = 1.0 / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
   n[0] *= rl; n[1] *= rl; n[2] *= rl;

   os << " facet normal " << n[0] << ' ' << n[1] << ' ' << n[2]
      << "\n  outer loop"
      << "\n   vertex " << p1[0] << ' ' << p1[1] << ' ' << p1[2]
      << "\n   vertex " << p2[0] << ' ' << p2[1] << ' ' << p2[2]
      << "\n   vertex " << p3[0] << ' ' << p3[1] << ' ' << p3[2]
      << "\n  endloop\n endfacet\n";
}

void GridFunction::SaveSTL(std::ostream &os, int TimesToRefine)
{
   Mesh *mesh = fes->GetMesh();

   if (mesh->Dimension() != 2)
   {
      return;
   }

   int i, j, k, l, n;
   DenseMatrix pointmat;
   Vector values;
   RefinedGeometry * RefG;
   real_t pts[4][3], bbox[3][2];

   os << "solid GridFunction\n";

   bbox[0][0] = bbox[0][1] = bbox[1][0] = bbox[1][1] =
                                             bbox[2][0] = bbox[2][1] = 0.0;
   for (i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementBaseGeometry(i);
      RefG = GlobGeometryRefiner.Refine(geom, TimesToRefine);
      GetValues(i, RefG->RefPts, values, pointmat);
      Array<int> &RG = RefG->RefGeoms;
      n = Geometries.NumBdr(geom);
      for (k = 0; k < RG.Size()/n; k++)
      {
         for (j = 0; j < n; j++)
         {
            l = RG[n*k+j];
            pts[j][0] = pointmat(0,l);
            pts[j][1] = pointmat(1,l);
            pts[j][2] = values(l);
         }

         if (n == 3)
         {
            SaveSTLTri(os, pts[0], pts[1], pts[2]);
         }
         else
         {
            SaveSTLTri(os, pts[0], pts[1], pts[2]);
            SaveSTLTri(os, pts[0], pts[2], pts[3]);
         }
      }

      if (i == 0)
      {
         bbox[0][0] = pointmat(0,0);
         bbox[0][1] = pointmat(0,0);
         bbox[1][0] = pointmat(1,0);
         bbox[1][1] = pointmat(1,0);
         bbox[2][0] = values(0);
         bbox[2][1] = values(0);
      }

      for (j = 0; j < values.Size(); j++)
      {
         if (bbox[0][0] > pointmat(0,j))
         {
            bbox[0][0] = pointmat(0,j);
         }
         if (bbox[0][1] < pointmat(0,j))
         {
            bbox[0][1] = pointmat(0,j);
         }
         if (bbox[1][0] > pointmat(1,j))
         {
            bbox[1][0] = pointmat(1,j);
         }
         if (bbox[1][1] < pointmat(1,j))
         {
            bbox[1][1] = pointmat(1,j);
         }
         if (bbox[2][0] > values(j))
         {
            bbox[2][0] = values(j);
         }
         if (bbox[2][1] < values(j))
         {
            bbox[2][1] = values(j);
         }
      }
   }

   mfem::out << "[xmin,xmax] = [" << bbox[0][0] << ',' << bbox[0][1] << "]\n"
             << "[ymin,ymax] = [" << bbox[1][0] << ',' << bbox[1][1] << "]\n"
             << "[zmin,zmax] = [" << bbox[2][0] << ',' << bbox[2][1] << ']'
             << endl;

   os << "endsolid GridFunction" << endl;
}

std::ostream &operator<<(std::ostream &os, const GridFunction &sol)
{
   sol.Save(os);
   return os;
}

void GridFunction::LegacyNCReorder()
{
   const Mesh* mesh = fes->GetMesh();
   MFEM_ASSERT(mesh->Nonconforming(), "");

   // get the mapping (old_vertex_index -> new_vertex_index)
   Array<int> new_vertex, old_vertex;
   mesh->ncmesh->LegacyToNewVertexOrdering(new_vertex);
   MFEM_ASSERT(new_vertex.Size() == mesh->GetNV(), "");

   // get the mapping (new_vertex_index -> old_vertex_index)
   old_vertex.SetSize(new_vertex.Size());
   for (int i = 0; i < new_vertex.Size(); i++)
   {
      old_vertex[new_vertex[i]] = i;
   }

   Vector tmp = *this;

   // reorder vertex DOFs
   Array<int> old_vdofs, new_vdofs;
   for (int i = 0; i < mesh->GetNV(); i++)
   {
      fes->GetVertexVDofs(i, old_vdofs);
      fes->GetVertexVDofs(new_vertex[i], new_vdofs);

      for (int j = 0; j < new_vdofs.Size(); j++)
      {
         tmp(new_vdofs[j]) = (*this)(old_vdofs[j]);
      }
   }

   // reorder edge DOFs -- edge orientation has changed too
   Array<int> dofs, ev;
   for (int i = 0; i < mesh->GetNEdges(); i++)
   {
      mesh->GetEdgeVertices(i, ev);
      if (old_vertex[ev[0]] > old_vertex[ev[1]])
      {
         const int *ind = fes->FEColl()->DofOrderForOrientation(Geometry::SEGMENT, -1);

         fes->GetEdgeInteriorDofs(i, dofs);
         for (int k = 0; k < dofs.Size(); k++)
         {
            int new_dof = dofs[k];
            int old_dof = dofs[(ind[k] < 0) ? -1-ind[k] : ind[k]];

            for (int j = 0; j < fes->GetVDim(); j++)
            {
               int new_vdof = fes->DofToVDof(new_dof, j);
               int old_vdof = fes->DofToVDof(old_dof, j);

               real_t sign = (ind[k] < 0) ? -1.0 : 1.0;
               tmp(new_vdof) = sign * (*this)(old_vdof);
            }
         }
      }
   }

   Vector::Swap(tmp);
}

real_t ZZErrorEstimator(BilinearFormIntegrator &blfi,
                        GridFunction &u,
                        GridFunction &flux, Vector &error_estimates,
                        Array<int>* aniso_flags,
                        int with_subdomains,
                        bool with_coeff)
{
   FiniteElementSpace *ufes = u.FESpace();
   FiniteElementSpace *ffes = flux.FESpace();
   ElementTransformation *Transf;

   int dim = ufes->GetMesh()->Dimension();
   int nfe = ufes->GetNE();

   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl, fla, d_xyz;

   error_estimates.SetSize(nfe);
   if (aniso_flags)
   {
      aniso_flags->SetSize(nfe);
      d_xyz.SetSize(dim);
   }

   int nsd = 1;
   if (with_subdomains)
   {
      nsd = ufes->GetMesh()->attributes.Max();
   }

   real_t total_error = 0.0;
   for (int s = 1; s <= nsd; s++)
   {
      // This calls the parallel version when u is a ParGridFunction
      u.ComputeFlux(blfi, flux, with_coeff, (with_subdomains ? s : -1));

      for (int i = 0; i < nfe; i++)
      {
         if (with_subdomains && ufes->GetAttribute(i) != s) { continue; }

         const DofTransformation* const utrans = ufes->GetElementVDofs(i, udofs);
         const DofTransformation* const ftrans = ffes->GetElementVDofs(i, fdofs);

         u.GetSubVector(udofs, ul);
         flux.GetSubVector(fdofs, fla);
         if (utrans)
         {
            utrans->InvTransformPrimal(ul);
         }
         if (ftrans)
         {
            ftrans->InvTransformPrimal(fla);
         }

         Transf = ufes->GetElementTransformation(i);
         blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                                 *ffes->GetFE(i), fl, with_coeff);

         fl -= fla;

         real_t eng = blfi.ComputeFluxEnergy(*ffes->GetFE(i), *Transf, fl,
                                             (aniso_flags ? &d_xyz : NULL));

         error_estimates(i) = std::sqrt(eng);
         total_error += eng;

         if (aniso_flags)
         {
            real_t sum = 0;
            for (int k = 0; k < dim; k++)
            {
               sum += d_xyz[k];
            }

            real_t thresh = 0.15 * 3.0/dim;
            int flag = 0;
            for (int k = 0; k < dim; k++)
            {
               if (d_xyz[k] / sum > thresh) { flag |= (1 << k); }
            }

            (*aniso_flags)[i] = flag;
         }
      }
   }
#ifdef MFEM_USE_MPI
   auto pfes = dynamic_cast<ParFiniteElementSpace*>(ufes);
   if (pfes)
   {
      auto process_local_error = total_error;
      MPI_Allreduce(&process_local_error, &total_error, 1,
                    MPITypeMap<real_t>::mpi_type,
                    MPI_SUM, pfes->GetComm());
   }
#endif // MFEM_USE_MPI
   return std::sqrt(total_error);
}

void TensorProductLegendre(int dim,                // input
                           int order,              // input
                           const Vector &x_in,     // input
                           const Vector &xmax,     // input
                           const Vector &xmin,     // input
                           Vector &poly,           // output
                           real_t angle,           // input (optional)
                           const Vector *midpoint) // input (optional)
{
   MFEM_VERIFY(dim >= 1, "dim must be positive");
   MFEM_VERIFY(dim <= 3, "dim cannot be greater than 3");
   MFEM_VERIFY(order >= 0, "order cannot be negative");

   bool rotate = (angle != 0.0) || (midpoint->Norml2() != 0.0);

   Vector x(dim);
   if (rotate && dim == 2)
   {
      // Rotate coordinates to match rotated bounding box
      Vector tmp(dim);
      tmp = x_in;
      tmp -= *midpoint;
      x[0] = tmp[0]*cos(-angle) - tmp[1]*sin(-angle);
      x[1] = tmp[0]*sin(-angle) + tmp[1]*cos(-angle);
   }
   else
   {
      // Bounding box is not reoriented no need to change orientation
      x = x_in;
   }

   // Map x to [0, 1] to use CalcLegendre since it uses shifted Legendre Polynomials.
   real_t x1 = (x(0) - xmin(0))/(xmax(0)-xmin(0)), x2, x3;
   Vector poly_x(order+1), poly_y(order+1), poly_z(order+1);
   poly1d.CalcLegendre(order, x1, poly_x.GetData());
   if (dim > 1)
   {
      x2 = (x(1)-xmin(1))/(xmax(1)-xmin(1));
      poly1d.CalcLegendre(order, x2, poly_y.GetData());
   }
   if (dim == 3)
   {
      x3 = (x(2)-xmin(2))/(xmax(2)-xmin(2));
      poly1d.CalcLegendre(order, x3, poly_z.GetData());
   }

   int basis_dimension = static_cast<int>(pow(order+1,dim));
   poly.SetSize(basis_dimension);
   switch (dim)
   {
      case 1:
      {
         for (int i = 0; i <= order; i++)
         {
            poly(i) = poly_x(i);
         }
      }
      break;
      case 2:
      {
         for (int j = 0; j <= order; j++)
         {
            for (int i = 0; i <= order; i++)
            {
               int cnt = i + (order+1) * j;
               poly(cnt) = poly_x(i) * poly_y(j);
            }
         }
      }
      break;
      case 3:
      {
         for (int k = 0; k <= order; k++)
         {
            for (int j = 0; j <= order; j++)
            {
               for (int i = 0; i <= order; i++)
               {
                  int cnt = i + (order+1) * j + (order+1) * (order+1) * k;
                  poly(cnt) = poly_x(i) * poly_y(j) * poly_z(k);
               }
            }
         }
      }
      break;
      default:
      {
         MFEM_ABORT("TensorProductLegendre: invalid value of dim");
      }
   }
}

void BoundingBox(const Array<int> &patch,  // input
                 FiniteElementSpace *ufes, // input
                 int order,                // input
                 Vector &xmin,             // output
                 Vector &xmax,             // output
                 real_t &angle,            // output
                 Vector &midpoint,         // output
                 int iface)                // input (optional)
{
   Mesh *mesh = ufes->GetMesh();
   int dim = mesh->Dimension();
   int num_elems = patch.Size();
   IsoparametricTransformation Tr;

   xmax = -infinity();
   xmin = infinity();
   angle = 0.0;
   midpoint = 0.0;
   bool rotate = (dim == 2);

   // Rotate bounding box to match the face orientation
   if (rotate && iface >= 0)
   {
      IntegrationPoint reference_pt;
      mesh->GetFaceTransformation(iface, &Tr);
      Vector physical_pt(2);
      Vector physical_diff(2);
      physical_diff = 0.0;
      // Get the endpoints of the edge in physical space
      // then compute midpoint and angle
      for (int i = 0; i < 2; i++)
      {
         reference_pt.Set1w((real_t)i, 0.0);
         Tr.Transform(reference_pt, physical_pt);
         midpoint += physical_pt;
         physical_pt *= pow(-1.0,i);
         physical_diff += physical_pt;
      }
      midpoint /= 2.0;
      angle = atan2(physical_diff(1),physical_diff(0));
   }

   for (int i = 0; i < num_elems; i++)
   {
      int ielem = patch[i];
      const IntegrationRule *ir = &(IntRules.Get(mesh->GetElementGeometry(ielem),
                                                 order));
      ufes->GetElementTransformation(ielem, &Tr);
      for (int k = 0; k < ir->GetNPoints(); k++)
      {
         const IntegrationPoint ip = ir->IntPoint(k);
         Vector transip(dim);
         Tr.Transform(ip, transip);
         if (rotate)
         {
            transip -= midpoint;
            Vector tmp(dim);
            tmp = transip;
            transip[0] = tmp[0]*cos(-angle) - tmp[1]*sin(-angle);
            transip[1] = tmp[0]*sin(-angle) + tmp[1]*cos(-angle);
         }
         for (int d = 0; d < dim; d++) { xmax(d) = max(xmax(d), transip(d)); }
         for (int d = 0; d < dim; d++) { xmin(d) = min(xmin(d), transip(d)); }
      }
   }
}

real_t LSZZErrorEstimator(BilinearFormIntegrator &blfi,  // input
                          GridFunction &u,               // input
                          Vector &error_estimates,       // output
                          bool subdomain_reconstruction, // input (optional)
                          bool with_coeff,               // input (optional)
                          real_t tichonov_coeff)         // input (optional)
{
   MFEM_VERIFY(tichonov_coeff >= 0.0, "tichonov_coeff cannot be negative");
   FiniteElementSpace *ufes = u.FESpace();
   ElementTransformation *Transf;

   Mesh *mesh = ufes->GetMesh();
   int dim = mesh->Dimension();
   int sdim = mesh->SpaceDimension();
   int nfe = ufes->GetNE();
   int nfaces = ufes->GetNF();

   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl, fla;

   error_estimates.SetSize(nfe);
   error_estimates = 0.0;
   Array<int> counters(nfe);
   counters = 0;

   Vector xmax(dim);
   Vector xmin(dim);
   real_t angle = 0.0;
   Vector midpoint(dim);

   // Compute the number of subdomains
   int nsd = 1;
   if (subdomain_reconstruction)
   {
      nsd = ufes->GetMesh()->attributes.Max();
   }

   real_t total_error = 0.0;
   for (int iface = 0; iface < nfaces; iface++)
   {
      // 1.A. Find all elements in the face patch.
      int el1;
      int el2;
      mesh->GetFaceElements(iface, &el1, &el2);
      Array<int> patch(2);
      patch[0] = el1; patch[1] = el2;

      // 1.B. Check if boundary face or non-conforming coarse face and continue if true.
      if (el1 == -1 || el2 == -1)
      {
         continue;
      }

      // 1.C Check if face patch crosses an attribute interface and
      // continue if true (only active if subdomain_reconstruction == true)
      if (nsd > 1)
      {
         int el1_attr = ufes->GetAttribute(el1);
         int el2_attr = ufes->GetAttribute(el2);
         if (el1_attr != el2_attr) { continue; }
      }

      // 2. Compute global flux polynomial.

      // 2.A. Compute polynomial order of patch (for hp FEM)
      const int patch_order = max(ufes->GetElementOrder(el1),
                                  ufes->GetElementOrder(el2));

      int num_basis_functions = static_cast<int>(pow(patch_order+1,dim));
      int flux_order = 2*patch_order + 1;
      DenseMatrix A(num_basis_functions);
      Array<real_t> b(sdim * num_basis_functions);
      A = 0.0;
      b = 0.0;

      // 2.B. Estimate the smallest bounding box around the face patch
      //      (this is used in 2.C.ii. to define a global polynomial basis)
      BoundingBox(patch, ufes, flux_order,
                  xmin, xmax, angle, midpoint, iface);

      // 2.C. Compute the normal equations for the least-squares problem
      // 2.C.i. Evaluate the discrete flux at all integration points in all
      //        elements in the face patch
      for (int i = 0; i < patch.Size(); i++)
      {
         int ielem = patch[i];
         const IntegrationRule *ir = &(IntRules.Get(mesh->GetElementGeometry(ielem),
                                                    flux_order));
         int num_integration_pts = ir->GetNPoints();

         const DofTransformation* const utrans = ufes->GetElementVDofs(ielem, udofs);
         u.GetSubVector(udofs, ul);
         if (utrans)
         {
            utrans->InvTransformPrimal(ul);
         }
         Transf = ufes->GetElementTransformation(ielem);
         FiniteElement *dummy = nullptr;
         blfi.ComputeElementFlux(*ufes->GetFE(ielem), *Transf, ul,
                                 *dummy, fl, with_coeff, ir);

         // 2.C.ii. Use global polynomial basis to construct normal
         //         equations
         for (int k = 0; k < num_integration_pts; k++)
         {
            const IntegrationPoint ip = ir->IntPoint(k);
            real_t tmp[3];
            Vector transip(tmp, 3);
            Transf->Transform(ip, transip);

            Vector p;
            TensorProductLegendre(dim, patch_order, transip, xmax, xmin, p, angle,
                                  &midpoint);
            AddMultVVt(p, A);

            for (int l = 0; l < num_basis_functions; l++)
            {
               // Loop through each component of the discrete flux
               for (int n = 0; n < sdim; n++)
               {
                  b[l + n * num_basis_functions] += p(l) * fl(k + n * num_integration_pts);
               }
            }
         }
      }

      // 2.D. Shift spectrum of A to avoid conditioning issues.
      //      Regularization is necessary if the tensor product space used for the
      //      flux reconstruction leads to an underdetermined system of linear equations.
      //      This should not happen if there are tensor product elements in the patch,
      //      but it can happen if there are other element shapes (those with few
      //      integration points) in the patch.
      for (int i = 0; i < num_basis_functions; i++)
      {
         A(i,i) += tichonov_coeff;
      }

      // 2.E. Solve for polynomial coefficients
      Array<int> ipiv(num_basis_functions);
      LUFactors lu(A.Data(), ipiv);
      real_t TOL = 1e-9;
      if (!lu.Factor(num_basis_functions,TOL))
      {
         // Singular matrix
         mfem::out << "LSZZErrorEstimator: Matrix A is singular.\t"
                   << "Consider increasing tichonov_coeff." << endl;
         for (int i = 0; i < num_basis_functions; i++)
         {
            A(i,i) += 1e-8;
         }
         lu.Factor(num_basis_functions,TOL);
      }
      lu.Solve(num_basis_functions, sdim, b);

      // 2.F. Construct l2-minimizing global polynomial
      auto global_poly_tmp = [=] (const Vector &x, Vector &f)
      {
         Vector p;
         TensorProductLegendre(dim, patch_order, x, xmax, xmin, p, angle, &midpoint);
         f = 0.0;
         for (int i = 0; i < num_basis_functions; i++)
         {
            for (int j = 0; j < sdim; j++)
            {
               f(j) += b[i + j * num_basis_functions] * p(i);
            }
         }
      };
      VectorFunctionCoefficient global_poly(sdim, global_poly_tmp);

      // 3. Compute error contributions from the face.
      real_t element_error = 0.0;
      real_t patch_error = 0.0;
      for (int i = 0; i < patch.Size(); i++)
      {
         int ielem = patch[i];
         element_error = u.ComputeElementGradError(ielem, &global_poly);
         element_error *= element_error;
         patch_error += element_error;
         error_estimates(ielem) += element_error;
         counters[ielem]++;
      }

      total_error += patch_error;
   }

   // 4. Calibrate the final error estimates. Note that the l2 norm of
   //    error_estimates vector converges to total_error.
   //    The error estimates have been calibrated so that high order
   //    benchmark problems with tensor product elements are asymptotically
   //    exact.
   for (int ielem = 0; ielem < nfe; ielem++)
   {
      if (counters[ielem] == 0)
      {
         error_estimates(ielem) = infinity();
      }
      else
      {
         error_estimates(ielem) /= counters[ielem]/2.0;
         error_estimates(ielem) = sqrt(error_estimates(ielem));
      }
   }
   return std::sqrt(total_error/dim);
}

real_t ComputeElementLpDistance(real_t p, int i,
                                GridFunction& gf1, GridFunction& gf2)
{
   real_t norm = 0.0;

   FiniteElementSpace *fes1 = gf1.FESpace();
   FiniteElementSpace *fes2 = gf2.FESpace();

   const FiniteElement* fe1 = fes1->GetFE(i);
   const FiniteElement* fe2 = fes2->GetFE(i);

   const IntegrationRule *ir;
   int intorder = 2*std::max(fe1->GetOrder(),fe2->GetOrder()) + 1; // <-------
   ir = &(IntRules.Get(fe1->GetGeomType(), intorder));
   int nip = ir->GetNPoints();
   Vector val1, val2;

   ElementTransformation *T = fes1->GetElementTransformation(i);
   for (int j = 0; j < nip; j++)
   {
      const IntegrationPoint &ip = ir->IntPoint(j);
      T->SetIntPoint(&ip);

      gf1.GetVectorValue(i, ip, val1);
      gf2.GetVectorValue(i, ip, val2);

      val1 -= val2;
      real_t errj = val1.Norml2();
      if (p < infinity())
      {
         errj = pow(errj, p);
         norm += ip.weight * T->Weight() * errj;
      }
      else
      {
         norm = std::max(norm, errj);
      }
   }

   if (p < infinity())
   {
      // Negative quadrature weights may cause the norm to be negative
      norm = pow(fabs(norm), 1./p);
   }

   return norm;
}


real_t ExtrudeCoefficient::Eval(ElementTransformation &T,
                                const IntegrationPoint &ip)
{
   ElementTransformation *T_in =
      mesh_in->GetElementTransformation(T.ElementNo / n);
   T_in->SetIntPoint(&ip);
   return sol_in.Eval(*T_in, ip);
}


GridFunction *Extrude1DGridFunction(Mesh *mesh, Mesh *mesh2d,
                                    GridFunction *sol, const int ny)
{
   GridFunction *sol2d;

   FiniteElementCollection *solfec2d;
   const char *name = sol->FESpace()->FEColl()->Name();
   string cname = name;
   if (cname == "Linear")
   {
      solfec2d = new LinearFECollection;
   }
   else if (cname == "Quadratic")
   {
      solfec2d = new QuadraticFECollection;
   }
   else if (cname == "Cubic")
   {
      solfec2d = new CubicFECollection;
   }
   else if (!strncmp(name, "H1_", 3))
   {
      solfec2d = new H1_FECollection(atoi(name + 7), 2);
   }
   else if (!strncmp(name, "H1Pos_", 6))
   {
      // use regular (nodal) H1_FECollection
      solfec2d = new H1_FECollection(atoi(name + 10), 2);
   }
   else if (!strncmp(name, "L2_T", 4))
   {
      solfec2d = new L2_FECollection(atoi(name + 10), 2);
   }
   else if (!strncmp(name, "L2_", 3))
   {
      solfec2d = new L2_FECollection(atoi(name + 7), 2);
   }
   else if (!strncmp(name, "L2Int_", 6))
   {
      solfec2d = new L2_FECollection(atoi(name + 7), 2, BasisType::GaussLegendre,
                                     FiniteElement::INTEGRAL);
   }
   else
   {
      mfem::err << "Extrude1DGridFunction : unknown FE collection : "
                << cname << endl;
      return NULL;
   }
   FiniteElementSpace *solfes2d;
   // assuming sol is scalar
   solfes2d = new FiniteElementSpace(mesh2d, solfec2d);
   sol2d = new GridFunction(solfes2d);
   sol2d->MakeOwner(solfec2d);
   {
      GridFunctionCoefficient csol(sol);
      ExtrudeCoefficient c2d(mesh, csol, ny);
      sol2d->ProjectCoefficient(c2d);
   }
   return sol2d;
}

}
