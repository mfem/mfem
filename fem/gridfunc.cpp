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

// Implementation of GridFunction

#include "fem.hpp"
#include <cstring>
#include <math.h>

GridFunction::GridFunction(Mesh *m, istream &input)
   : Vector()
{
   const int bufflen = 256;
   char buff[bufflen];
   int vdim;

   input >> ws;
   input.getline(buff, bufflen);  // 'FiniteElementSpace'
   if (strcmp(buff, "FiniteElementSpace"))
      mfem_error("GridFunction::GridFunction():"
                 " input stream is not a GridFunction!");
   input.getline(buff, bufflen, ' '); // 'FiniteElementCollection:'
   input >> ws;
   input.getline(buff, bufflen);
   fec = FiniteElementCollection::New(buff);
   input.getline(buff, bufflen, ' '); // 'VDim:'
   input >> vdim;
   input.getline(buff, bufflen, ' '); // 'Ordering:'
   int ordering;
   input >> ordering;
   input.getline(buff, bufflen); // read the empty line
   fes = new FiniteElementSpace(m, fec, vdim, ordering);
   Vector::Load(input, fes->GetVSize());
}

GridFunction::GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces)
{
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
}

GridFunction::~GridFunction()
{
   if (fec)
   {
      delete fes;
      delete fec;
   }
}

void GridFunction::Update(FiniteElementSpace *f)
{
   if (fec)
   {
      delete fes;
      delete fec;
      fec = NULL;
   }
   fes = f;
   SetSize(fes->GetVSize());
}

void GridFunction::Update(FiniteElementSpace *f, Vector &v, int v_offset)
{
   if (fec)
   {
      delete fes;
      delete fec;
      fec = NULL;
   }
   fes = f;
   SetDataAndSize((double *)v + v_offset, fes->GetVSize());
}

int GridFunction::VectorDim() const
{
   const FiniteElement *fe = fes->GetFE(0);

   if (fe->GetRangeType() == FiniteElement::SCALAR)
      return fes->GetVDim();
   return fe->GetDim();
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
   fes->GetFE(i)->CalcShape(ip, DofVal);
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
      int dim = FElem->GetDim();
      DenseMatrix vshape(dof, dim);
      ElementTransformation *Tr = fes->GetElementTransformation(i);
      Tr->SetIntPoint(&ip);
      FElem->CalcVShape(*Tr, vshape);
      val.SetSize(dim);
      vshape.MultTranspose(loc_data, val);
   }
}

void GridFunction::GetValues(int i, const IntegrationRule &ir, Vector &vals,
                             DenseMatrix &tr, int vdim)
   const
{
   Array<int> dofs;

   int k, n;

   n = ir.GetNPoints();
   vals.SetSize(n);
   fes->GetElementVDofs(i, dofs);
   const FiniteElement *FElem = fes->GetFE(i);
   ElementTransformation *ET;
   ET = fes->GetElementTransformation(i);
   ET->Transform(ir, tr);
   int dof = FElem->GetDof();
   Vector DofVal(dof);
   vdim--;
   for (k = 0; k < n; k++)
   {
      FElem->CalcShape(ir.IntPoint(k), DofVal);
      vals(k) = 0.0;
      for (int j = 0; j < dof; j++)
         if (dofs[dof*vdim+j] >= 0)
            vals(k) += DofVal(j) * data[dofs[dof*vdim+j]];
         else
            vals(k) -= DofVal(j) * data[-1-dofs[dof*vdim+j]];
   }
}

int GridFunction::GetFaceValues(int i, int side, const IntegrationRule &ir,
                                Vector &vals, DenseMatrix &tr,
                                int vdim) const
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
         di = 0;
      else
         di = 1;
   }
   else
      di = side;
   if (di == 0)
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

   return di;
}

void GridFunction::GetVectorValues(int i, const IntegrationRule &ir,
                                   DenseMatrix &vals, DenseMatrix &tr)
   const
{
   const FiniteElement *FElem = fes->GetFE(i);
   int dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   int nip = ir.GetNPoints();
   ElementTransformation *Tr = fes->GetElementTransformation(i);
   Tr->Transform(ir, tr);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
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
      int dim = FElem->GetDim();
      DenseMatrix vshape(dof, dim);
      vals.SetSize(dim, nip);
      Vector val_j;
      for (int j = 0; j < nip; j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         Tr->SetIntPoint(&ip);
         FElem->CalcVShape(*Tr, vshape);
         vals.GetColumnReference(j, val_j);
         vshape.MultTranspose(loc_data, val_j);
      }
   }
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
         di = 0;
      else
         di = 1;
   }
   else
      di = side;
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

void GridFunction::GetValuesFrom(GridFunction &orig_func)
{
   // Without averaging ...

   FiniteElementSpace *orig_fes = orig_func.FESpace();
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

void GridFunction::GetBdrValuesFrom(GridFunction &orig_func)
{
   // Without averaging ...

   FiniteElementSpace *orig_fes = orig_func.FESpace();
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

   int d, j, k, n, dim, dof, ind;

   n = ir.GetNPoints();
   fes->GetElementVDofs(i, vdofs);
   const FiniteElement *fe = fes->GetFE(i);
   dof = fe->GetDof();
   dim = fe->GetDim();
   int *dofs = &vdofs[comp*dof];
   transf = fes->GetElementTransformation(i);
   transf->Transform(ir, tr);
   vals.SetSize(n, dim);
   DenseMatrix vshape(dof, dim);
   double a;
   for (k = 0; k < n; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      transf->SetIntPoint(&ip);
      fe->CalcVShape(*transf, vshape);
      for (d = 0; d < dim; d++)
      {
         a = 0.0;
         for (j = 0; j < dof; j++)
            if ( (ind=dofs[j]) >= 0 )
               a += vshape(j, d) * data[ind];
            else
               a -= vshape(j, d) * data[-1-ind];
         vals(k, d) = a;
      }
   }
}

void GridFunction::ReorderByNodes()
{
   if (fes->GetOrdering() == Ordering::byNODES)
      return;

   int i, j, k;
   int vdim = fes->GetVDim();
   int ndofs = fes->GetNDofs();
   double *temp = new double[size];

   k = 0;
   for (j = 0; j < ndofs; j++)
      for (i = 0; i < vdim; i++)
         temp[j+i*ndofs] = data[k++];

   for (i = 0; i < size; i++)
      data[i] = temp[i];

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
      val(i) /= overlap[i];
}

void GridFunction::ProjectVectorFieldOn(GridFunction &vec_field, int comp)
{
   FiniteElementSpace *new_fes = vec_field.FESpace();

   int d, i, k, ind, dof;
   Array<int> overlap(new_fes->GetVSize());
   Array<int> new_vdofs;
   DenseMatrix vals, tr;

   overlap = 0;
   vec_field = 0.0;

   for (i = 0; i < new_fes->GetNE(); i++)
   {
      const FiniteElement *fe = new_fes->GetFE(i);
      const IntegrationRule &ir = fe->GetNodes();
      GetVectorFieldValues(i, ir, vals, tr, comp);
      new_fes->GetElementVDofs(i, new_vdofs);
      dof = fe->GetDof();
      for (d = 0; d < fe->GetDim(); d++)
         for (k = 0; k < dof; k++)
         {
            if ( (ind=new_vdofs[dof*d+k]) < 0 )
               ind = -1-ind, vals(k, d) = - vals(k, d);
            vec_field(ind) += vals(k, d);
            overlap[ind]++;
         }
   }

   for (i = 0; i < overlap.Size(); i++)
      vec_field(i) /= overlap[i];
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
      overlap[i] = 0;
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
            a += inv_jac(j, der_comp) * pt_grad(j);
         der(der_dofs[k]) += a;
         overlap[der_dofs[k]]++;
      }
   }

   for (i = 0; i < overlap.Size(); i++)
      der(i) /= overlap[i];
}


void GridFunction::GetVectorGradientHat(
   ElementTransformation &T, DenseMatrix &gh)
{
   int elNo = T.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   int dim = FElem->GetDim(), dof = FElem->GetDof();
   Array<int> vdofs;
   fes->GetElementVDofs(elNo, vdofs);
   Vector loc_data;
   GetSubVector(vdofs, loc_data);
   // assuming scalar FE
   DenseMatrix dshape(dof, dim);
   FElem->CalcDShape(T.GetIntPoint(), dshape);
   gh.SetSize(dim);
   for (int i = 0; i < dim; i++)
      for (int j = 0; j < dim; j++)
      {
         double gij = 0.0;
         for (int k = 0; k < dof; k++)
            gij += loc_data(i * dof + k) * dshape(k, j);
         gh(i, j) = gij;
      }
}

double GridFunction::GetDivergence(ElementTransformation &tr)
{
   double div_v;
   int elNo = tr.ElementNo;
   const FiniteElement *FElem = fes->GetFE(elNo);
   if (FElem->GetRangeType() == FiniteElement::SCALAR)
   {
      DenseMatrix grad_hat;
      GetVectorGradientHat(tr, grad_hat);
      int dim = grad_hat.Size();
      DenseMatrix Jinv(dim);
      CalcInverse(tr.Jacobian(), Jinv);
      div_v = 0.0;
      for (int i = 0; i < dim; i++)
         for (int j = 0; j < dim; j++)
            div_v += grad_hat(i, j) * Jinv(j, i);
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

void GridFunction::GetGradient(ElementTransformation &tr, Vector &grad)
{
   mfem_error("GridFunction::GetGradient(...) is not implemented!");
}

void GridFunction::GetGradients(const int elem, const IntegrationRule &ir,
                                DenseMatrix &grad)
{
   const FiniteElement *fe = fes->GetFE(elem);
   ElementTransformation *Tr = fes->GetElementTransformation(elem);
   DenseMatrix dshape(fe->GetDof(), fe->GetDim());
   DenseMatrix Jinv(fe->GetDim());
   Vector lval, gh(fe->GetDim()), gcol;
   Array<int> dofs;
   fes->GetElementDofs(elem, dofs);
   GetSubVector(dofs, lval);
   grad.SetSize(fe->GetDim(), ir.GetNPoints());
   for (int i = 0; i < ir.GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir.IntPoint(i);
      fe->CalcDShape(ip, dshape);
      dshape.MultTranspose(lval, gh);
      Tr->SetIntPoint(&ip);
      grad.GetColumnReference(i, gcol);
      CalcInverse(Tr->Jacobian(), Jinv);
      Jinv.MultTranspose(gh, gcol);
   }
}

void GridFunction::GetVectorGradient(
   ElementTransformation &tr, DenseMatrix &grad)
{
   DenseMatrix grad_hat;
   GetVectorGradientHat(tr, grad_hat);
   DenseMatrix Jinv(grad_hat.Size());
   CalcInverse(tr.Jacobian(), Jinv);
   grad.SetSize(grad_hat.Size());
   Mult(grad_hat, Jinv, grad);
}

void GridFunction::GetElementAverages(GridFunction &avgs)
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
      avgs(i) /= int_psi(i);
}

void GridFunction::GetNodalValues(Vector &nval, int vdim) const
{
   int i, j;
   Array<int> vertices;
   Array<double> values;
   Array<int> overlap(fes->GetNV());
   nval.SetSize(fes->GetNV());

   for (i = 0; i < overlap.Size(); i++)
   {
      nval(i) = 0.0;
      overlap[i] = 0;
   }
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
      nval(i) /= overlap[i];
}

void GridFunction::ProjectCoefficient(Coefficient &coeff)
{
   int i;
   Array<int> vdofs;
   Vector vals;

   DeltaCoefficient *delta_c = dynamic_cast<DeltaCoefficient *>(&coeff);

   if (delta_c == NULL)
   {
      for (i = 0; i < fes->GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         vals.SetSize(vdofs.Size());
         fes->GetFE(i)->Project(coeff, *fes->GetElementTransformation(i), vals);
         SetSubVector(vdofs, vals);
      }
   }
   else
   {
      Mesh *mesh = fes->GetMesh();
      const int dim = mesh->Dimension();
      const double *center = delta_c->Center();
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

      if (min_dist >= delta_c->Tol())
         return;

      // find the elements that have 'v_idx' as a vertex
      MassIntegrator Mi(*delta_c->Weight());
      DenseMatrix loc_mass;
      Array<int> vertices;
      Vector loc_mass_vals;
      double integral = 0.0;
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
               for (int k = 0; k < loc_mass_vals.Size(); k++)
                  integral += loc_mass_vals(k);
               break;
            }
      }

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

void GridFunction::ProjectCoefficient(Coefficient *coeff[])
{
   int i, j, fdof, d, ind;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;

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
         for (d = 0; d < fes->GetVDim(); d++)
         {
            val = coeff[d]->Eval(*transf, ip);
            if ( (ind = vdofs[fdof*d+j]) < 0 )
               val = -val, ind = -1-ind;
            (*this)(ind) = val;
         }
      }
   }
}

void GridFunction::ProjectBdrCoefficient(
   Coefficient *coeff[], Array<int> &attr)
{
   int i, j, fdof, d, ind;
   double val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;

   for (i = 0; i < fes->GetNBE(); i++)
   {
      if ( attr[fes->GetBdrAttribute(i)-1] )
      {
         fe = fes->GetBE(i);
         fdof = fe->GetDof();
         transf = fes->GetBdrElementTransformation(i);
         const IntegrationRule &ir = fe->GetNodes();
         fes->GetBdrElementVDofs(i, vdofs);
         for (j = 0; j < fdof; j++)
         {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);
            for (d = 0; d < fes->GetVDim(); d++)
            {
               val = coeff[d]->Eval(*transf, ip);
               if ( (ind = vdofs[fdof*d+j]) < 0 )
                  val = -val, ind = -1-ind;
               (*this)(ind) = val;
            }
         }
      }
   }
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
         continue;
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
         const DenseMatrix &J = T->Jacobian();
         if (dim == 2)
         {
            nor(0) =  J(1,0);
            nor(1) = -J(0,0);
         }
         else if (dim == 3)
         {
            nor(0) = J(1,0)*J(2,1) - J(2,0)*J(1,1);
            nor(1) = J(2,0)*J(0,1) - J(0,0)*J(2,1);
            nor(2) = J(0,0)*J(1,1) - J(1,0)*J(0,1);
         }
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
         continue;
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      const IntegrationRule &ir = fe->GetNodes();
      lvec.SetSize(fe->GetDof());
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         T->SetIntPoint(&ip);
         vcoeff.Eval(vc, *T, ip);
         const DenseMatrix &J = T->Jacobian();
         if (dim == 2)
         {
            nor(0) =  J(1,0);
            nor(1) = -J(0,0);
         }
         else if (dim == 3)
         {
            nor(0) = J(1,0)*J(2,1) - J(2,0)*J(1,1);
            nor(1) = J(2,0)*J(0,1) - J(0,0)*J(2,1);
            nor(2) = J(0,0)*J(1,1) - J(1,0)*J(0,1);
         }
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
   const FiniteElement *fe;
   ElementTransformation *T;
   Array<int> dofs;
   Vector lvec;

   for (int i = 0; i < fes->GetNBE(); i++)
   {
      if (bdr_attr[fes->GetBdrAttribute(i)-1] == 0)
         continue;
      fe = fes->GetBE(i);
      T = fes->GetBdrElementTransformation(i);
      fes->GetBdrElementDofs(i, dofs);
      lvec.SetSize(fe->GetDof());
      fe->Project(vcoeff, *T, lvec);
      SetSubVector(dofs, lvec);
   }
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
         ir = irs[fe->GetGeomType()];
      else
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
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
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               else
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
            transf->SetIntPoint(&ip);
            a -= exsol[d]->Eval(*transf, ip);
            error += ip.weight * transf->Weight() * a * a;
         }
      }
   }

   if (error < 0.0)
      return -sqrt(-error);
   return sqrt(error);
}

double GridFunction::ComputeL2Error(
   VectorCoefficient &exsol, const IntegrationRule *irs[],
   Array<int> *elems) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals, tr;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      if (elems != NULL && (*elems)[i] == 0)  continue;
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
         ir = irs[fe->GetGeomType()];
      else
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      GetVectorValues(i, *ir, vals, tr);
      T = fes->GetElementTransformation(i);
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
      return -sqrt(-error);
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
               el_dofs(k) =   (*this)(vdofs[k]);
            else
               el_dofs(k) = - (*this)(-1-vdofs[k]);
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
               intorder = k;
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
               el_dofs(k) =   (*this)(vdofs[k]);
            else
               el_dofs(k) = - (*this)(-1-vdofs[k]);
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
                  el_dofs(k) =   (*this)(vdofs[k]);
               else
                  el_dofs(k) = - (*this)(-1-vdofs[k]);
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
      return -sqrt(-error);
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
         ir = irs[fe->GetGeomType()];
      else
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
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
                  a += (*this)(vdofs[fdof*d+k]) * shape(k);
               else
                  a -= (*this)(-1-vdofs[fdof*d+k]) * shape(k);
            a -= exsol[d]->Eval(*transf, ip);
            a = fabs(a);
            if (error < a)
               error = a;
         }
      }
   }

   return error;
}

double GridFunction::ComputeMaxError(
   VectorCoefficient &exsol, const IntegrationRule *irs[]) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals, tr;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
         ir = irs[fe->GetGeomType()];
      else
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      GetVectorValues(i, *ir, vals, tr);
      T = fes->GetElementTransformation(i);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      // compute the lengths of the errors at the integration points
      // thus the vector max. norm is rotationally invariant
      vals.Norm2(loc_errs);
      double loc_error = loc_errs.Normlinf();
      if (error < loc_error)
         error = loc_error;
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
         if (elems != NULL && (*elems)[i] == 0)  continue;
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = fes->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         shape.SetSize(fdof);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
            ir = irs[fe->GetGeomType()];
         else
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
               el_dofs(k) = (*this)(vdofs[k]);
            else
               el_dofs(k) = -(*this)(-1-vdofs[k]);
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
         if (elems != NULL && (*elems)[i] == 0)  continue;
         fe = fes->GetFE(i);
         fdof = fe->GetDof();
         transf = mesh->GetElementTransformation(i);
         el_dofs.SetSize(fdof);
         dshape.SetSize(fdof, dim);
         dshapet.SetSize(fdof, dim);
         intorder = 2*fe->GetOrder() + 1; // <----------
         const IntegrationRule *ir;
         if (irs)
            ir = irs[fe->GetGeomType()];
         else
            ir = &(IntRules.Get(fe->GetGeomType(), intorder));
         fes->GetElementVDofs(i, vdofs);
         for (k = 0; k < fdof; k++)
            if (vdofs[k] >= 0)
               el_dofs(k) = (*this)(vdofs[k]);
            else
               el_dofs(k) = -(*this)(-1-vdofs[k]);
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

double GridFunction::ComputeL1Error(
   VectorCoefficient &exsol, const IntegrationRule *irs[]) const
{
   double error = 0.0;
   const FiniteElement *fe;
   ElementTransformation *T;
   DenseMatrix vals, exact_vals, tr;
   Vector loc_errs;

   for (int i = 0; i < fes->GetNE(); i++)
   {
      fe = fes->GetFE(i);
      int intorder = 2*fe->GetOrder() + 1; // <----------
      const IntegrationRule *ir;
      if (irs)
         ir = irs[fe->GetGeomType()];
      else
         ir = &(IntRules.Get(fe->GetGeomType(), intorder));
      GetVectorValues(i, *ir, vals, tr);
      T = fes->GetElementTransformation(i);
      exsol.Eval(exact_vals, *T, *ir);
      vals -= exact_vals;
      loc_errs.SetSize(vals.Width());
      // compute the lengths of the errors at the integration points
      // thus the vector L_1 norm is rotationally invariant
      vals.Norm2(loc_errs);
      for (int j = 0; j < ir->GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         T->SetIntPoint(&ip);
         error += ip.weight * T->Weight() * loc_errs(j);
      }
   }

   return error;
}

GridFunction & GridFunction::operator=(double value)
{
   for (int i = 0; i < size; i++)
      data[i] = value;
   return *this;
}

GridFunction & GridFunction::operator=(const Vector &v)
{
   for (int i = 0; i < size; i++)
      data[i] = v(i);
   return *this;
}

GridFunction & GridFunction::operator=(const GridFunction &v)
{
   return this->operator=((const Vector &)v);
}

void GridFunction::Save(ostream &out)
{
   fes->Save(out);
   out << '\n';
   if (fes->GetOrdering() == Ordering::byNODES)
      Vector::Print(out, 1);
   else
      Vector::Print(out, fes->GetVDim());
}

void GridFunction::SaveVTK(ostream &out, const string &field_name, int ref)
{
   Mesh *mesh = fes->GetMesh();
   RefinedGeometry *RefG;
   Vector val;
   DenseMatrix vval, pmat;

   if (VectorDim() == 1)
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
   else
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
               out << 0.0;
            else
               out << vval(2, j);
            out << '\n';
         }
      }
   }
}

void GridFunction::SaveSTLTri(ostream &out, double p1[], double p2[],
                              double p3[])
{
   double v1[3] = { p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
   double v2[3] = { p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2] };
   double n[] = {  v1[1] * v2[2] - v1[2] * v2[1],
                   v1[2] * v2[0] - v1[0] * v2[2],
                   v1[0] * v2[1] - v1[1] * v2[0]  };
   double rl = 1.0 / sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
   n[0] *= rl; n[1] *= rl; n[2] *= rl;

   out << " facet normal " << n[0] << ' ' << n[1] << ' ' << n[2]
       << "\n  outer loop"
       << "\n   vertex " << p1[0] << ' ' << p1[1] << ' ' << p1[2]
       << "\n   vertex " << p2[0] << ' ' << p2[1] << ' ' << p2[2]
       << "\n   vertex " << p3[0] << ' ' << p3[1] << ' ' << p3[2]
       << "\n  endloop\n endfacet\n";
}

void GridFunction::SaveSTL(ostream &out, int TimesToRefine)
{
   Mesh *mesh = fes->GetMesh();

   if (mesh->Dimension() != 2)
      return;

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
      n = fes->GetFE(i)->GetGeomType();
      RefG = GlobGeometryRefiner.Refine(n, TimesToRefine);
      GetValues(i, RefG->RefPts, values, pointmat);
      Array<int> &RG = RefG->RefGeoms;
      n = Geometries.NumBdr(n);
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
            bbox[0][0] = pointmat(0,j);
         if (bbox[0][1] < pointmat(0,j))
            bbox[0][1] = pointmat(0,j);
         if (bbox[1][0] > pointmat(1,j))
            bbox[1][0] = pointmat(1,j);
         if (bbox[1][1] < pointmat(1,j))
            bbox[1][1] = pointmat(1,j);
         if (bbox[2][0] > values(j))
            bbox[2][0] = values(j);
         if (bbox[2][1] < values(j))
            bbox[2][1] = values(j);
      }
   }

   cout << "[xmin,xmax] = [" << bbox[0][0] << ',' << bbox[0][1] << "]\n"
        << "[ymin,ymax] = [" << bbox[1][0] << ',' << bbox[1][1] << "]\n"
        << "[zmin,zmax] = [" << bbox[2][0] << ',' << bbox[2][1] << ']'
        << endl;

   out << "endsolid GridFunction" << endl;
}


void ComputeFlux(BilinearFormIntegrator &blfi,
                 GridFunction &u,
                 GridFunction &flux, int wcoef, int sd)
{
   int i, j, nfe;
   FiniteElementSpace *ufes, *ffes;
   ElementTransformation *Transf;

   ufes = u.FESpace();
   ffes = flux.FESpace();
   nfe = ufes->GetNE();
   Array<int> udofs;
   Array<int> fdofs;
   Array<int> overlap(flux.Size());
   Vector ul, fl;

   flux = 0.0;

   for (i = 0; i < overlap.Size(); i++)
      overlap[i] = 0;

   for (i = 0; i < nfe; i++)
      if (sd < 0 || ufes->GetAttribute(i) == sd)
      {
         ufes->GetElementVDofs(i, udofs);
         ffes->GetElementVDofs(i, fdofs);

         ul.SetSize(udofs.Size());
         for (j = 0; j < ul.Size(); j++)
            ul(j) = u(udofs[j]);

         Transf = ufes->GetElementTransformation(i);
         blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                                 *ffes->GetFE(i), fl, wcoef);

         flux.AddElementVector(fdofs, fl);

         for (j = 0; j < fdofs.Size(); j++)
            overlap[fdofs[j]]++;
      }

   for (i = 0; i < overlap.Size(); i++)
      if (overlap[i] != 0)
         flux(i) /= overlap[i];
}

void ZZErrorEstimator(BilinearFormIntegrator &blfi,
                      GridFunction &u,
                      GridFunction &flux, Vector &ErrorEstimates,
                      int wsd)
{
   int i, j, s, nfe, nsd;
   FiniteElementSpace *ufes, *ffes;
   ElementTransformation *Transf;

   ufes = u.FESpace();
   ffes = flux.FESpace();
   nfe = ufes->GetNE();
   Array<int> udofs;
   Array<int> fdofs;
   Vector ul, fl, fla;

   ErrorEstimates.SetSize(nfe);

   nsd = 1;
   if (wsd)
      for (i = 0; i < nfe; i++)
         if ( (j=ufes->GetAttribute(i)) > nsd)
            nsd = j;

   for (s = 1; s <= nsd; s++)
   {
      if (wsd)
         ComputeFlux(blfi, u, flux, 0, s);
      else
         ComputeFlux(blfi, u, flux, 0);

      for (i = 0; i < nfe; i++)
         if (!wsd || ufes->GetAttribute(i) == s)
         {
            ufes->GetElementVDofs(i, udofs);
            ffes->GetElementVDofs(i, fdofs);

            ul.SetSize(udofs.Size());
            for (j = 0; j < ul.Size(); j++)
               ul(j) = u(udofs[j]);

            fla.SetSize(fdofs.Size());
            for (j = 0; j < fla.Size(); j++)
               fla(j) = flux(fdofs[j]);

            Transf = ufes->GetElementTransformation(i);
            blfi.ComputeElementFlux(*ufes->GetFE(i), *Transf, ul,
                                    *ffes->GetFE(i), fl, 0);

            fl -= fla;

            ErrorEstimates(i) = blfi.ComputeFluxEnergy(*ffes->GetFE(i),
                                                       *Transf, fl);
         }
   }
}
