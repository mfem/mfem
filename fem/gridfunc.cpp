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

// Implementation of GridFunction

#include "gridfunc.hpp"
#include "../mesh/nurbs.hpp"
#include "../general/text.hpp"

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
   fec = fes->Load(m, input);

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
   }
   sequence = fes->GetSequence();
}

GridFunction::GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces)
{
   UseDevice(true);

   // all GridFunctions must have the same FE collection, vdim, ordering
   int vdim, ordering;

   fes = gf_array[0]->FESpace();
   fec = FiniteElementCollection::New(fes->FEColl()->Name());
   vdim = fes->GetVDim();
   ordering = fes->GetOrdering();
   fes = new FiniteElementSpace(m, fec, vdim, ordering);
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
      const double *l_data = gf_array[i]->GetData();
      double *g_data = data;
      if (ordering == Ordering::byNODES)
      {
         for (int d = 0; d < vdim; d++)
         {
            memcpy(g_data+vi, l_data, l_nvdofs*sizeof(double));
            l_data += l_nvdofs;
            g_data += g_nvdofs;
            memcpy(g_data+ei, l_data, l_nedofs*sizeof(double));
            l_data += l_nedofs;
            g_data += g_nedofs;
            memcpy(g_data+fi, l_data, l_nfdofs*sizeof(double));
            l_data += l_nfdofs;
            g_data += g_nfdofs;
            memcpy(g_data+di, l_data, l_nddofs*sizeof(double));
            l_data += l_nddofs;
            g_data += g_nddofs;
         }
      }
      else
      {
         memcpy(g_data+vdim*vi, l_data, vdim*l_nvdofs*sizeof(double));
         l_data += vdim*l_nvdofs;
         g_data += vdim*g_nvdofs;
         memcpy(g_data+vdim*ei, l_data, vdim*l_nedofs*sizeof(double));
         l_data += vdim*l_nedofs;
         g_data += vdim*g_nedofs;
         memcpy(g_data+vdim*fi, l_data, vdim*l_nfdofs*sizeof(double));
         l_data += vdim*l_nfdofs;
         g_data += vdim*g_nfdofs;
         memcpy(g_data+vdim*di, l_data, vdim*l_nddofs*sizeof(double));
         l_data += vdim*l_nddofs;
         g_data += vdim*g_nddofs;
      }
      vi += l_nvdofs;
      ei += l_nedofs;
      fi += l_nfdofs;
      di += l_nddofs;
   }
   sequence = 0;
}

void GridFunction::Destroy()
{
   if (fec)
   {
      delete fes;
      delete fec;
      fec = NULL;
   }
}

void GridFunction::Update()
{
   if (fes->GetSequence() == sequence)
   {
      return; // space and grid function are in sync, no-op
   }
   if (fes->GetSequence() != sequence + 1)
   {
      MFEM_ABORT("Error in update sequence. GridFunction needs to be updated "
                 "right after the space is updated.");
   }
   sequence = fes->GetSequence();

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
}

void GridFunction::SetSpace(FiniteElementSpace *f)
{
   if (f != fes) { Destroy(); }
   fes = f;
   SetSize(fes->GetVSize());
   sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, double *v)
{
   if (f != fes) { Destroy(); }
   fes = f;
   NewDataAndSize(v, fes->GetVSize());
   sequence = fes->GetSequence();
}

void GridFunction::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   MFEM_ASSERT(v.Size() >= v_offset + f->GetVSize(), "");
   if (f != fes) { Destroy(); }
   fes = f;
   v.UseDevice(true);
   NewMemoryAndSize(Memory<double>(v.GetMemory(), v_offset, fes->GetVSize()),
                    fes->GetVSize(), true);
   sequence = fes->GetSequence();
}

void GridFunction::MakeTRef(FiniteElementSpace *f, double *tv)
{
   if (!f->GetProlongationMatrix())
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
   if (!f->GetProlongationMatrix())
   {
      MakeRef(f, tv, tv_offset);
      t_vec.NewMemoryAndSize(data, size, false);
   }
   else
   {
      MFEM_ASSERT(tv.Size() >= tv_offset + f->GetTrueVSize(), "");
      SetSpace(f); // works in parallel
      tv.UseDevice(true);
      const int tv_size = f->GetTrueVSize();
      t_vec.NewMemoryAndSize(Memory<double>(tv.GetMemory(), tv_offset, tv_size),
                             tv_size, true);
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

      ufes->GetElementVDofs(i, udofs);
      ffes->GetElementVDofs(i, fdofs);

      u.GetSubVector(udofs, ul);

      Transf = ufes->GetElementTransformation(i);
      blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                              *ffes->GetFE(i), fl, wcoef);

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
      const FiniteElementCollection *fec = fes->FEColl();
      static const Geometry::Type geoms[3] =
      { Geometry::SEGMENT, Geometry::TRIANGLE, Geometry::TETRAHEDRON };
      fe = fec->FiniteElementForGeometry(geoms[fes->GetMesh()->Dimension()-1]);
   }
   else
   {
      fe = fes->GetFE(0);
   }
   if (!fe || fe->GetRangeType() == FiniteElement::SCALAR)
   {
      return fes->GetVDim();
   }
   return fes->GetVDim()*fes->GetMesh()->SpaceDimension();
}

void GridFunction::GetTrueDofs(Vector &tv) const
{
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   if (!R)
   {
      // R is identity -> make tv a reference to *this
      tv.NewDataAndSize(const_cast<double*>((const double*)data), size);
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
      if (tv.GetData() != data)
      {
         *this = tv;
      }
   }
   else
   {
      cP->Mult(tv, *this);
   }
}

void GridFunction::GetNodalValues(int i, Array<double> &nval, int vdim) const
{
   Array<int> vdofs;

   int k;

   fes->GetElementVDofs(i, vdofs);
   const FiniteElement *FElem = fes->GetFE(i);
   const IntegrationRule *ElemVert =
      Geometries.GetVertices(FElem->GetGeomType());
   int dof = FElem->GetDof();
   int n = ElemVert->GetNPoints();
   nval.SetSize(n);
   vdim--;
   Vector loc_data;
   GetSubVector(vdofs, loc_data);

   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      for (k = 0; k < n; k++)
      {
         FElem->CalcShape(ElemVert->IntPoint(k), shape);
         nval[k] = shape * ((const double *)loc_data + dof * vdim);
      }
   }
   else
   {
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      DenseMatrix vshape(dof, FElem->GetDim());
      for (k = 0; k < n; k++)
      {
         Tr->SetIntPoint(&ElemVert->IntPoint(k));
         FElem->CalcVShape(*Tr, vshape);
         nval[k] = loc_data * (&vshape(0,vdim));
      }
   }
}

double GridFunction::GetValue(int i, const IntegrationPoint &ip, int vdim)
const
{
   Array<int> dofs;
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   Vector DofVal(dofs.Size()), LocVec;
   const FiniteElement *fe = fes->GetFE(i);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   fe->CalcShape(ip, DofVal);
   GetSubVector(dofs, LocVec);

   return (DofVal * LocVec);
}

void GridFunction::GetVectorValue(int i, const IntegrationPoint &ip,
                                  Vector &val) const
{
   const FiniteElement *FElem = fes->GetFE(i);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      FElem->CalcShape(ip, shape);
      int vdim = fes->GetVDim();
      val.SetSize(vdim);
      for (int k = 0; k < vdim; k++)
      {
         val(k) = shape * ((const double *)loc_data + dof * k);
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      DenseMatrix vshape(dof, spaceDim);
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      Tr->SetIntPoint(&ip);
      FElem->CalcVShape(*Tr, vshape);
      val.SetSize(spaceDim);
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
   fes->GetElementDofs(i, dofs);
   fes->DofsToVDofs(vdim-1, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");
   int dof = FElem->GetDof();
   Vector DofVal(dof), loc_data(dof);
   GetSubVector(dofs, loc_data);
   for (int k = 0; k < n; k++)
   {
      FElem->CalcShape(ir.IntPoint(k), DofVal);
      vals(k) = DofVal * loc_data;
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

      for (int i = 0; i < size; i++)
      {
         for (int d = 0; d < dof; d++)
         {
            hess(k,i) += DofHes(d,i) * loc_data[d];
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

void GridFunction::GetVectorValues(ElementTransformation &T,
                                   const IntegrationRule &ir,
                                   DenseMatrix &vals) const
{
   const FiniteElement *FElem = fes->GetFE(T.ElementNo);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(T.ElementNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   int nip = ir.GetNPoints();
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      Vector shape(dof);
      int vdim = fes->GetVDim();
      vals.SetSize(vdim, nip);
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         FElem->CalcShape(ip, shape);
         for (int k = 0; k < vdim; k++)
         {
            vals(k,j) = shape * ((const double *)loc_data + dof * k);
         }
      }
   }
   else
   {
      int spaceDim = fes->GetMesh()->SpaceDimension();
      DenseMatrix vshape(dof, spaceDim);
      vals.SetSize(spaceDim, nip);
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

void GridFunction::GetVectorValues(int i, const IntegrationRule &ir,
                                   DenseMatrix &vals, DenseMatrix &tr) const
{
   ElementTransformation *Tr = fes->GetElementTransformation(i);
   Tr->Transform(ir, tr);

   GetVectorValues(*Tr, ir, vals);
}

int GridFunction::GetFaceVectorValues(
   int i, int side, const IntegrationRule &ir,
   DenseMatrix &vals, DenseMatrix &tr) const
{
   int n, di;
   FaceElementTransformations *Transf;

   n = ir.GetNPoints();
   IntegrationRule eir(n);  // ---
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
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 4);
      Transf->Loc1.Transform(ir, eir);
      GetVectorValues(Transf->Elem1No, eir, vals, tr);
   }
   else
   {
      Transf = fes->GetMesh()->GetFaceElementTransformations(i, 8);
      Transf->Loc2.Transform(ir, eir);
      GetVectorValues(Transf->Elem2No, eir, vals, tr);
   }

   return di;
}

void GridFunction::GetValuesFrom(const GridFunction &orig_func)
{
   // Without averaging ...

   const FiniteElementSpace *orig_fes = orig_func.FESpace();
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, orig_loc_values;
   int i, j, d, ne, dof, odof, vdim;

   ne = fes->GetNE();
   vdim = fes->GetVDim();
   for (i = 0; i < ne; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      orig_fes->GetElementVDofs(i, orig_vdofs);
      orig_func.GetSubVector(orig_vdofs, orig_loc_values);
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
            loc_values(d*dof+j) =
               shape * ((const double *)orig_loc_values + d * odof) ;
         }
      }
      SetSubVector(vdofs, loc_values);
   }
}

void GridFunction::GetBdrValuesFrom(const GridFunction &orig_func)
{
   // Without averaging ...

   const FiniteElementSpace *orig_fes = orig_func.FESpace();
   Array<int> vdofs, orig_vdofs;
   Vector shape, loc_values, orig_loc_values;
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
            loc_values(d*dof+j) =
               shape * ((const double *)orig_loc_values + d * odof);
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

   int d, j, k, n, sdim, dof, ind;

   n = ir.GetNPoints();
   fes->GetElementVDofs(i, vdofs);
   const FiniteElement *fe = fes->GetFE(i);
   dof = fe->GetDof();
   sdim = fes->GetMesh()->SpaceDimension();
   int *dofs = &vdofs[comp*dof];
   transf = fes->GetElementTransformation(i);
   transf->Transform(ir, tr);
   vals.SetSize(n, sdim);
   DenseMatrix vshape(dof, sdim);
   double a;
   for (k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      transf->SetIntPoint(&ip);
      fe->CalcVShape(*transf, vshape);
      for (d = 0; d < sdim; d++)
      {
         a = 0.0;
         for (j = 0; j < dof; j++)
            if ( (ind=dofs[j]) >= 0 )
            {
               a += vshape(j, d) * data[ind];
            }
            else
            {
               a -= vshape(j, d) * data[-1-ind];
            }
         vals(k, d) = a;
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
   double *temp = new double[size];

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

void GridFunction::GetDerivative(int comp, int der_comp, GridFunction &der)
{
   FiniteElementSpace * der_fes = der.FESpace();
   ElementTransformation * transf;
   Array<int> overlap(der_fes->GetVSize());
   Array<int> der_dofs, vdofs;
   DenseMatrix dshape, inv_jac;
   Vector pt_grad, loc_func;
   int i, j, k, dim, dof, der_dof, ind;
   double a;

   for (i = 0; i < overlap.Size(); i++)
   {
      overlap[i] = 0;
   }
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
         overlap[der_dofs[k]]++;
      }
   }

   for (i = 0; i < overlap.Size(); i++)
   {
      der(i) /= overlap[i];
   }
}


void GridFunction::GetVectorGradientHat(
   ElementTransformation &T, DenseMatrix &gh) const
{
   int elNo = T.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   int dim = FElem->GetDim(), dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(elNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   // assuming scalar FE
   int vdim = fes->GetVDim();
   DenseMatrix dshape(dof, dim);
   FElem->CalcDShape(T.GetIntPoint(), dshape);
   gh.SetSize(vdim, dim);
   DenseMatrix loc_data_mat(loc_data.GetData(), dof, vdim);
   MultAtB(loc_data_mat, dshape, gh);
}

double GridFunction::GetDivergence(ElementTransformation &tr) const
{
   double div_v;
   int elNo = tr.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      DenseMatrix grad_hat;
      GetVectorGradientHat(tr, grad_hat);
      const DenseMatrix &Jinv = tr.InverseJacobian();
      div_v = 0.0;
      for (int i = 0; i < Jinv.Width(); i++)
      {
         for (int j = 0; j < Jinv.Height(); j++)
         {
            div_v += grad_hat(i, j) * Jinv(j, i);
         }
      }
   }
   else
   {
      // Assuming RT-type space
      Array<int> dofs;
      fes->GetElementDofs(elNo, dofs);
      Vector loc_data, divshape(FElem->GetDof());
      GetSubVector(dofs, loc_data);
      FElem->CalcDivShape(tr.GetIntPoint(), divshape);
      div_v = (loc_data * divshape) / tr.Weight();
   }
   return div_v;
}

void GridFunction::GetCurl(ElementTransformation &tr, Vector &curl) const
{
   int elNo = tr.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      MFEM_ASSERT(FElem->GetMapType() == FiniteElement::VALUE,
                  "invalid FE map type");
      DenseMatrix grad_hat;
      GetVectorGradientHat(tr, grad_hat);
      const DenseMatrix &Jinv = tr.InverseJacobian();
      DenseMatrix grad(grad_hat.Height(), Jinv.Width()); // vdim x FElem->Dim
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
      fes->GetElementDofs(elNo, dofs);
      Vector loc_data;
      GetSubVector(dofs, loc_data);
      DenseMatrix curl_shape(FElem->GetDof(), FElem->GetDim() == 3 ? 3 : 1);
      FElem->CalcCurlShape(tr.GetIntPoint(), curl_shape);
      curl.SetSize(curl_shape.Width());
      if (curl_shape.Width() == 3)
      {
         double curl_hat[3];
         curl_shape.MultTranspose(loc_data, curl_hat);
         tr.Jacobian().Mult(curl_hat, curl);
      }
      else
      {
         curl_shape.MultTranspose(loc_data, curl);
      }
      curl /= tr.Weight();
   }
}

void GridFunction::GetGradient(ElementTransformation &tr, Vector &grad) const
{
   int elNo = tr.ElementNo;
   const FiniteElement *fe = fes->GetFE(elNo);
   MFEM_ASSERT(fe->GetMapType() == FiniteElement::VALUE, "invalid FE map type");
   int dim = fe->GetDim(), dof = fe->GetDof();
   DenseMatrix dshape(dof, dim);
   Vector lval, gh(dim);
   Array<int> dofs;

   grad.SetSize(dim);
   fes->GetElementDofs(elNo, dofs);
   GetSubVector(dofs, lval);
   fe->CalcDShape(tr.GetIntPoint(), dshape);
   dshape.MultTranspose(lval, gh);
   tr.InverseJacobian().MultTranspose(gh, grad);
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
   Array<int> dofs;
   fes->GetElementDofs(elNo, dofs);
   GetSubVector(dofs, lval);
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
   ElementTransformation &tr, DenseMatrix &grad) const
{
   MFEM_ASSERT(fes->GetFE(tr.ElementNo)->GetMapType() == FiniteElement::VALUE,
               "invalid FE map type");
   DenseMatrix grad_hat;
   GetVectorGradientHat(tr, grad_hat);
   const DenseMatrix &Jinv = tr.InverseJacobian();
   grad.SetSize(grad_hat.Height(), Jinv.Width());
   Mult(grad_hat, Jinv, grad);
}

void GridFunction::GetElementAverages(GridFunction &avgs) const
{
   MassIntegrator Mi;
   DenseMatrix loc_mass;
   Array<int> te_dofs, tr_dofs;
   Vector loc_avgs, loc_this;
   Vector int_psi(avgs.Size());

   avgs = 0.0;
   int_psi = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
      Mi.AssembleElementMatrix2(*fes->GetFE(i), *avgs.FESpace()->GetFE(i),
                                *fes->GetElementTransformation(i), loc_mass);
      fes->GetElementDofs(i, tr_dofs);
      avgs.FESpace()->GetElementDofs(i, te_dofs);
      GetSubVector(tr_dofs, loc_this);
      loc_avgs.SetSize(te_dofs.Size());
      loc_mass.Mult(loc_this, loc_avgs);
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

      src.fes->GetElementVDofs(i, src_vdofs);
      src.GetSubVector(src_vdofs, src_lvec);
      for (int vd = 0; vd < vdim; vd++)
      {
         P.Mult(&src_lvec[vd*P.Width()], &dest_lvec[vd*P.Height()]);
      }
      fes->GetElementVDofs(i, dest_vdofs);
      SetSubVector(dest_vdofs, dest_lvec);
   }
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                const Vector &_lo, const Vector &_hi)
{
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);
   GetSubVector(vdofs, vals);

   MFEM_ASSERT(weights.Size() == size, "Different # of weights and dofs.");
   MFEM_ASSERT(_lo.Size() == size, "Different # of lower bounds and dofs.");
   MFEM_ASSERT(_hi.Size() == size, "Different # of upper bounds and dofs.");

   int max_iter = 30;
   double tol = 1.e-12;
   SLBQPOptimizer slbqp;
   slbqp.SetMaxIter(max_iter);
   slbqp.SetAbsTol(1.0e-18);
   slbqp.SetRelTol(tol);
   slbqp.SetBounds(_lo, _hi);
   slbqp.SetLinearConstraint(weights, weights * vals);
   slbqp.SetPrintLevel(0); // print messages only if not converged
   slbqp.Mult(vals, new_vals);

   SetSubVector(vdofs, new_vals);
}

void GridFunction::ImposeBounds(int i, const Vector &weights,
                                double _min, double _max)
{
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   int size = vdofs.Size();
   Vector vals, new_vals(size);
   GetSubVector(vdofs, vals);

   double max_val = vals.Max();
   double min_val = vals.Min();

   if (max_val <= _min)
   {
      new_vals = _min;
      SetSubVector(vdofs, new_vals);
      return;
   }

   if (_min <= min_val && max_val <= _max)
   {
      return;
   }

   Vector minv(size), maxv(size);
   minv = (_min > min_val) ? _min : min_val;
   maxv = (_max < max_val) ? _max : max_val;

   ImposeBounds(i, weights, minv, maxv);
}

void GridFunction::GetNodalValues(Vector &nval, int vdim) const
{
   int i, j;
   Array<int> vertices;
   Array<double> values;
   Array<int> overlap(fes->GetNV());
   nval.SetSize(fes->GetNV());

   nval = 0.0;
   overlap = 0;
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
   Coefficient *coeff[], VectorCoefficient *vcoeff, Array<int> &attr,
   Array<int> &values_counter)
{
   int i, j, fdof, d, ind, vdim;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;
   Vector vc;

   values_counter.SetSize(Size());
   values_counter = 0;

   vdim = fes->GetVDim();
   for (i = 0; i < fes->GetNBE(); i++)
   {
      if (attr[fes->GetBdrAttribute(i) - 1] == 0) { continue; }

      fe = fes->GetBE(i);
      fdof = fe->GetDof();
      transf = fes->GetBdrElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      fes->GetBdrElementVDofs(i, vdofs);

      for (j = 0; j < fdof; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         transf->SetIntPoint(&ip);
         if (vcoeff) { vcoeff->Eval(vc, *transf, ip); }
         for (d = 0; d < vdim; d++)
         {
            if (!vcoeff && !coeff[d]) { continue; }

            val = vcoeff ? vc(d) : coeff[d]->Eval(*transf, ip);
            if ( (ind = vdofs[fdof*d+j]) < 0 )
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
   // (depend on) internal faces/elements. We use the virtual method
   // GetBoundaryClosure from NCMesh to resolve the dependencies.

   if (fes->Nonconforming() && fes->GetMesh()->Dimension() == 3)
   {
      Vector vals;
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices;
      ncmesh->GetBoundaryClosure(attr, bdr_vertices, bdr_edges);

      for (i = 0; i < bdr_edges.Size(); i++)
      {
         int edge = bdr_edges[i];
         fes->GetEdgeVDofs(edge, vdofs);
         if (vdofs.Size() == 0) { continue; }

         transf = mesh->GetEdgeTransformation(edge);
         transf->Attribute = -1; // TODO: set the boundary attribute
         fe = fes->GetEdgeElement(edge);
         if (!vcoeff)
         {
            vals.SetSize(fe->GetDof());
            for (d = 0; d < vdim; d++)
            {
               if (!coeff[d]) { continue; }

               fe->Project(*coeff[d], *transf, vals);
               for (int k = 0; k < vals.Size(); k++)
               {
                  ind = vdofs[d*vals.Size()+k];
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
            vals.SetSize(vdim*fe->GetDof());
            fe->Project(*vcoeff, *transf, vals);
            for (int k = 0; k < vals.Size(); k++)
            {
               ind = vdofs[k];
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
   }
}

static void accumulate_dofs(const Array<int> &dofs, const Vector &vals,
                            Vector &gf, Array<int> &values_counter)
{
   for (int i = 0; i < dofs.Size(); i++)
   {
      int k = dofs[i];
      double val = vals(i);
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
   VectorCoefficient &vcoeff, Array<int> &bdr_attr,
   Array<int> &values_counter)
{
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   Vector lvec;

   values_counter.SetSize(Size());
   values_counter = 0;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
      {
         continue;
      }
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      fes->GetBdrElementDofs(i, dofs);
      lvec.SetSize(fe->GetDof());
      fe->Project(vcoeff, *T, lvec);
      accumulate_dofs(dofs, lvec, *this, values_counter);
   }

   if (fes->Nonconforming() && fes->GetMesh()->Dimension() == 3)
   {
      Mesh *mesh = fes->GetMesh();
      NCMesh *ncmesh = mesh->ncmesh;
      Array<int> bdr_edges, bdr_vertices;
      ncmesh->GetBoundaryClosure(bdr_attr, bdr_vertices, bdr_edges);

      for (int i = 0; i < bdr_edges.Size(); i++)
      {
         int edge = bdr_edges[i];
         fes->GetEdgeDofs(edge, dofs);
         if (dofs.Size() == 0) { continue; }

         T = mesh->GetEdgeTransformation(edge);
         T->Attribute = -1; // TODO: set the boundary attribute
         fe = fes->GetEdgeElement(edge);
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
         MFEM_ABORT("invalud AvgType");
   }
}

void GridFunction::ProjectDeltaCoefficient(DeltaCoefficient &delta_coeff,
                                           double &integral)
{
   if (!fes->GetNE())
   {
      integral = 0.0;
      return;
   }

   Mesh *mesh = fes->GetMesh();
   const int dim = mesh->Dimension();
   const double *center = delta_coeff.Center();
   const double *vert = mesh->GetVertex(0);
   double min_dist, dist;
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
            fes->GetElementVDofs(i, vdofs);
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

   if (delta_c == NULL)
   {
      Array<int> vdofs;
      Vector vals;

      for (int i = 0; i < fes->GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
         SetSubVector(vdofs, vals);
      }
   }
   else
   {
      double integral;

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
   int i;
   Array<int> vdofs;
   Vector vals;

   for (i = 0; i < fes->GetNE(); i++)
   {
      fes->GetElementVDofs(i, vdofs);
      vals.SetSize(vdofs.Size());
      fes->GetFE(i)->Project(vcoeff, *fes->GetElementTransformation(i), vals);
      SetSubVector(vdofs, vals);
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

void GridFunction::ProjectCoefficient(Coefficient *coeff[])
{
   int i, j, fdof, d, ind, vdim;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;

   vdim = fes->GetVDim();
   for (i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      fdof = fe->GetDof();
      transf = fes->GetElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
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
                                         Array<int> &attr)
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

void GridFunction::ProjectBdrCoefficient(Coefficient *coeff[], Array<int> &attr)
{
   Array<int> values_counter;
   this->HostReadWrite();
   AccumulateAndCountBdrValues(coeff, NULL, attr, values_counter);
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

void GridFunction::ProjectBdrCoefficientNormal(
   VectorCoefficient &vcoeff, Array<int> &bdr_attr)
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
      fes->GetBdrElementDofs(i, dofs);
      SetSubVector(dofs, lvec);
   }
#endif
}

void GridFunction::ProjectBdrCoefficientTangent(
   VectorCoefficient &vcoeff, Array<int> &bdr_attr)
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

double GridFunction::ComputeL2Error(
   Coefficient *exsol[], const IntegrationRule *irs[]) const
{
   double error = 0.0, a;
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
      fes->GetElementVDofs(i, vdofs);
      for (j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         fe->CalcShape(ip, shape);
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
            transf->SetIntPoint(&ip);
            a -= exsol[d]->Eval(*transf, ip);
            error += ip.weight * transf->Weight() * a * a;
         }
      }
   }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeL2Error(
   VectorCoefficient &exsol, const IntegrationRule *irs[],
   Array<int> *elems) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0) { continue; }
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
      {
         ir = irs[fe->GetGeomType()];
      }
      else
      {
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
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
         error += ip.weight * T->Weight() * (loc_errs(j) * loc_errs(j));
      }
   }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeH1Error(
   Coefficient *exsol, VectorCoefficient *exgrad,
   Coefficient *ell_coeff, double Nu, int norm_type) const
{
   // assuming vdim is 1
   int i, fdof, dim, intorder, j, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   FaceElementTransformations *face_elem_transf;
   Vector e_grad, a_grad, shape, el_dofs, err_val, ell_coeff_val;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   IntegrationPoint eip;
   double error = 0.0;

   mesh = fes->GetMesh();
   dim = mesh->Dimension();
   e_grad.SetSize(dim);
   a_grad.SetSize(dim);
   Jinv.SetSize(dim);

   if (norm_type & 1)
      for (i = 0; i < mesh->GetNE(); i++)
      {
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = mesh->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         dshape.SetSize(fdof, dim);
         dshapet.SetSize(fdof, dim);
         intorder = 2 * fe->GetOrder(); // <----------
         const IntegrationRule &ir = IntRules.Get(fe->GetGeomType(), intorder);
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
            {
               el_dofs(k) =   (*this)(vdofs[k]);
            }
            else
            {
               el_dofs(k) = - (*this)(-1-vdofs[k]);
            }
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            fe->CalcDShape(ip, dshape);
            transf->SetIntPoint(&ip);
            exgrad->Eval(e_grad, *transf, ip);
            CalcInverse(transf->Jacobian(), Jinv);
            Mult(dshape, Jinv, dshapet);
            dshapet.MultTranspose(el_dofs, a_grad);
            e_grad -= a_grad;
            error += (ip.weight * transf->Weight() *
                      ell_coeff->Eval(*transf, ip) *
                      (e_grad * e_grad));
         }
      }

   if (norm_type & 2)
      for (i = 0; i < mesh->GetNFaces(); i++)
      {
         face_elem_transf = mesh->GetFaceElementTransformations(i, 5);
         int i1 = face_elem_transf->Elem1No;
         int i2 = face_elem_transf->Elem2No;
         intorder = fes->GetFE(i1)->GetOrder();
         if (i2 >= 0)
            if ( (k = fes->GetFE(i2)->GetOrder()) > intorder )
            {
               intorder = k;
            }
         intorder = 2 * intorder;  // <-------------
         const IntegrationRule &ir =
            IntRules.Get(face_elem_transf->FaceGeom, intorder);
         err_val.SetSize(ir.GetNPoints());
         ell_coeff_val.SetSize(ir.GetNPoints());
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
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            face_elem_transf->Loc1.Transform(ir.IntPoint(j), eip);
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
            for (j = 0; j < ir.GetNPoints(); j++)
            {
               face_elem_transf->Loc2.Transform(ir.IntPoint(j), eip);
               fe->CalcShape(eip, shape);
               transf->SetIntPoint(&eip);
               ell_coeff_val(j) += ell_coeff->Eval(*transf, eip);
               ell_coeff_val(j) *= 0.5;
               err_val(j) -= (exsol->Eval(*transf, eip) - (shape * el_dofs));
            }
         }
         face_elem_transf = mesh->GetFaceElementTransformations(i, 16);
         transf = face_elem_transf->Face;
         for (j = 0; j < ir.GetNPoints(); j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);
            error += (ip.weight * Nu * ell_coeff_val(j) *
                      pow(transf->Weight(), 1.0-1.0/(dim-1)) *
                      err_val(j) * err_val(j));
         }
      }

   if (error < 0.0)
   {
      return -sqrt(-error);
   }
   return sqrt(error);
}

double GridFunction::ComputeMaxError(
   Coefficient *exsol[], const IntegrationRule *irs[]) const
{
   double error = 0.0, a;
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

double GridFunction::ComputeW11Error(
   Coefficient *exsol, VectorCoefficient *exgrad, int norm_type,
   Array<int> *elems, const IntegrationRule *irs[]) const
{
   // assuming vdim is 1
   int i, fdof, dim, intorder, j, k;
   Mesh *mesh;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Vector e_grad, a_grad, shape, el_dofs, err_val, ell_coeff_val;
   DenseMatrix dshape, dshapet, Jinv;
   Array<int> vdofs;
   double a, error = 0.0;

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
            error += ip.weight * transf->Weight() * fabs(a);
         }
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
            error += ip.weight * transf->Weight() * e_grad.Norml1();
         }
      }

   return error;
}

double GridFunction::ComputeLpError(const double p, Coefficient &exsol,
                                    Coefficient *weight,
                                    const IntegrationRule *irs[]) const
{
   double error = 0.0;
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
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      GetValues(i, *ir, vals);
      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error = std::max(error, err);
         }
      }
   }

   if (p < infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1./p);
      }
      else
      {
         error = pow(error, 1./p);
      }
   }

   return error;
}

void GridFunction::ComputeElementLpErrors(const double p, Coefficient &exsol,
                                          GridFunction &error,
                                          Coefficient *weight,
                                          const IntegrationRule *irs[]) const
{
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
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
      GetValues(i, *ir, vals);
      T = fes->GetElementTransformation(i);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = fabs(vals(j) - exsol.Eval(*T, ip));
         if (p < infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error[i] += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error[i] = std::max(error[i], err);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         if (error[i] < 0.)
         {
            error[i] = -pow(-error[i], 1./p);
         }
         else
         {
            error[i] = pow(error[i], 1./p);
         }
      }
   }
}

double GridFunction::ComputeLpError(const double p, VectorCoefficient &exsol,
                                    Coefficient *weight,
                                    VectorCoefficient *v_weight,
                                    const IntegrationRule *irs[]) const
{
   double error = 0.0;
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
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
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
            double err = 0.0;
            for (int d = 0; d < vals.Height(); d++)
            {
               err += vals(d,j)*exact_vals(d,j);
            }
            loc_errs(j) = fabs(err);
         }
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = loc_errs(j);
         if (p < infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error = std::max(error, err);
         }
      }
   }

   if (p < infinity())
   {
      // negative quadrature weights may cause the error to be negative
      if (error < 0.)
      {
         error = -pow(-error, 1./p);
      }
      else
      {
         error = pow(error, 1./p);
      }
   }

   return error;
}

void GridFunction::ComputeElementLpErrors(const double p,
                                          VectorCoefficient &exsol,
                                          GridFunction &error,
                                          Coefficient *weight,
                                          VectorCoefficient *v_weight,
                                          const IntegrationRule *irs[]) const
{
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
         int intorder = 2*fe->GetOrder() + 1; // <----------
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      }
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
            double err = 0.0;
            for (int d = 0; d < vals.Height(); d++)
            {
               err += vals(d,j)*exact_vals(d,j);
            }
            loc_errs(j) = fabs(err);
         }
      }
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         double err = loc_errs(j);
         if (p < infinity())
         {
            err = pow(err, p);
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error[i] += ip.weight * T->Weight() * err;
         }
         else
         {
            if (weight)
            {
               err *= weight->Eval(*T, ip);
            }
            error[i] = std::max(error[i], err);
         }
      }
      if (p < infinity())
      {
         // negative quadrature weights may cause the error to be negative
         if (error[i] < 0.)
         {
            error[i] = -pow(-error[i], 1./p);
         }
         else
         {
            error[i] = pow(error[i], 1./p);
         }
      }
   }
}

GridFunction & GridFunction::operator=(double value)
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

void GridFunction::Save(std::ostream &out) const
{
   fes->Save(out);
   out << '\n';
#if 0
   // Testing: write NURBS GridFunctions using "NURBS_patches" format.
   if (fes->GetNURBSext())
   {
      out << "NURBS_patches\n";
      fes->GetNURBSext()->PrintSolution(*this, out);
      out.flush();
      return;
   }
#endif
   if (fes->GetOrdering() == Ordering::byNODES)
   {
      Vector::Print(out, 1);
   }
   else
   {
      Vector::Print(out, fes->GetVDim());
   }
   out.flush();
}

void GridFunction::SaveVTK(std::ostream &out, const std::string &field_name,
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
      out << "SCALARS " << field_name << " double 1\n"
          << "LOOKUP_TABLE default\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         GetValues(i, RefG->RefPts, val, pmat);

         for (int j = 0; j < val.Size(); j++)
         {
            out << val(j) << '\n';
         }
      }
   }
   else if ( (vec_dim == 2 || vec_dim == 3) && mesh->SpaceDimension() > 1)
   {
      // vector data
      out << "VECTORS " << field_name << " double\n";
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         RefG = GlobGeometryRefiner.Refine(
                   mesh->GetElementBaseGeometry(i), ref, 1);

         GetVectorValues(i, RefG->RefPts, vval, pmat);

         for (int j = 0; j < vval.Width(); j++)
         {
            out << vval(0, j) << ' ' << vval(1, j) << ' ';
            if (vval.Height() == 2)
            {
               out << 0.0;
            }
            else
            {
               out << vval(2, j);
            }
            out << '\n';
         }
      }
   }
   else
   {
      // other data: save the components as separate scalars
      for (int vd = 0; vd < vec_dim; vd++)
      {
         out << "SCALARS " << field_name << vd << " double 1\n"
             << "LOOKUP_TABLE default\n";
         for (int i = 0; i < mesh->GetNE(); i++)
         {
            RefG = GlobGeometryRefiner.Refine(
                      mesh->GetElementBaseGeometry(i), ref, 1);

            GetValues(i, RefG->RefPts, val, pmat, vd + 1);

            for (int j = 0; j < val.Size(); j++)
            {
               out << val(j) << '\n';
            }
         }
      }
   }
   out.flush();
}

void GridFunction::SaveSTLTri(std::ostream &out, double p1[], double p2[],
                              double p3[])
{
   double v1[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
   double v2[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };
   double n[] = {  v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0]
                };
   double rl = 1.0 / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
   n[0] *= rl; n[1] *= rl; n[2] *= rl;

   out << " facet normal " << n[0] << ' ' << n[1] << ' ' << n[2]
       << "\n  outer loop"
       << "\n   vertex " << p1[0] << ' ' << p1[1] << ' ' << p1[2]
       << "\n   vertex " << p2[0] << ' ' << p2[1] << ' ' << p2[2]
       << "\n   vertex " << p3[0] << ' ' << p3[1] << ' ' << p3[2]
       << "\n  endloop\n endfacet\n";
}

void GridFunction::SaveSTL(std::ostream &out, int TimesToRefine)
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
   double pts[4][3], bbox[3][2];

   out << "solid GridFunction\n";

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
            SaveSTLTri(out, pts[0], pts[1], pts[2]);
         }
         else
         {
            SaveSTLTri(out, pts[0], pts[1], pts[2]);
            SaveSTLTri(out, pts[0], pts[2], pts[3]);
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

   out << "endsolid GridFunction" << endl;
}

std::ostream &operator<<(std::ostream &out, const GridFunction &sol)
{
   sol.Save(out);
   return out;
}


QuadratureFunction::QuadratureFunction(Mesh *mesh, std::istream &in)
{
   const char *msg = "invalid input stream";
   string ident;

   qspace = new QuadratureSpace(mesh, in);
   own_qspace = true;

   in >> ident; MFEM_VERIFY(ident == "VDim:", msg);
   in >> vdim;

   Load(in, vdim*qspace->GetSize());
}

QuadratureFunction & QuadratureFunction::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

QuadratureFunction & QuadratureFunction::operator=(const Vector &v)
{
   MFEM_ASSERT(qspace && v.Size() == qspace->GetSize(), "");
   Vector::operator=(v);
   return *this;
}

QuadratureFunction & QuadratureFunction::operator=(const QuadratureFunction &v)
{
   return this->operator=((const Vector &)v);
}

void QuadratureFunction::Save(std::ostream &out) const
{
   qspace->Save(out);
   out << "VDim: " << vdim << '\n'
       << '\n';
   Vector::Print(out, vdim);
   out.flush();
}

std::ostream &operator<<(std::ostream &out, const QuadratureFunction &qf)
{
   qf.Save(out);
   return out;
}


double ZZErrorEstimator(BilinearFormIntegrator &blfi,
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

   double total_error = 0.0;
   for (int s = 1; s <= nsd; s++)
   {
      // This calls the parallel version when u is a ParGridFunction
      u.ComputeFlux(blfi, flux, with_coeff, (with_subdomains ? s : -1));

      for (int i = 0; i < nfe; i++)
      {
         if (with_subdomains && ufes->GetAttribute(i) != s) { continue; }

         ufes->GetElementVDofs(i, udofs);
         ffes->GetElementVDofs(i, fdofs);

         u.GetSubVector(udofs, ul);
         flux.GetSubVector(fdofs, fla);

         Transf = ufes->GetElementTransformation(i);
         blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                                 *ffes->GetFE(i), fl, with_coeff);

         fl -= fla;

         double err = blfi.ComputeFluxEnergy(*ffes->GetFE(i), *Transf, fl,
                                             (aniso_flags ? &d_xyz : NULL));

         error_estimates(i) = std::sqrt(err);
         total_error += err;

         if (aniso_flags)
         {
            double sum = 0;
            for (int k = 0; k < dim; k++)
            {
               sum += d_xyz[k];
            }

            double thresh = 0.15 * 3.0/dim;
            int flag = 0;
            for (int k = 0; k < dim; k++)
            {
               if (d_xyz[k] / sum > thresh) { flag |= (1 << k); }
            }

            (*aniso_flags)[i] = flag;
         }
      }
   }

   return std::sqrt(total_error);
}


double ComputeElementLpDistance(double p, int i,
                                GridFunction& gf1, GridFunction& gf2)
{
   double norm = 0.0;

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
      double err = val1.Norml2();
      if (p < infinity())
      {
         err = pow(err, p);
         norm += ip.weight * T->Weight() * err;
      }
      else
      {
         norm = std::max(norm, err);
      }
   }

   if (p < infinity())
   {
      // Negative quadrature weights may cause the norm to be negative
      if (norm < 0.)
      {
         norm = -pow(-norm, 1./p);
      }
      else
      {
         norm = pow(norm, 1./p);
      }
   }

   return norm;
}


double ExtrudeCoefficient::Eval(ElementTransformation &T,
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
