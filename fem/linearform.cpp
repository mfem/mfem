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

// Implementation of class LinearForm

#include "fem.hpp"

namespace mfem
{

// Fake ElementTransformation class for Dirac delta integration
class FakeTransformation : public ElementTransformation
{
protected:
   virtual const DenseMatrix &EvalJacobian() { return dFdx; };

public:
   FakeTransformation(int dim)
   {
      Wght = 1.0;
      EvalState = WEIGHT_MASK;
      dFdx.Diag(1.0,dim);
   }
   virtual void Transform(const IntegrationPoint &, Vector &) {};
   virtual void Transform(const IntegrationRule &, DenseMatrix &) {};
   virtual void Transform(const DenseMatrix &matrix, DenseMatrix &result) { result = matrix; };
   virtual int Order() { return 0; }
   virtual int OrderJ() { return 0; }
   virtual int OrderW() { return 0; }
   virtual int OrderGrad(const FiniteElement *fe) { return 0; }
   virtual int GetSpaceDim() { return dFdx.Width(); }
   virtual int TransformBack(const Vector &, IntegrationPoint &) { return 2; }
};

void LinearForm::AddDomainIntegrator (LinearFormIntegrator * lfi)
{
   dlfi.Append (lfi);
   check_delta = true;
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi)
{
   blfi.Append (lfi);
}

void LinearForm::AddBdrFaceIntegrator (LinearFormIntegrator * lfi)
{
   flfi.Append(lfi);
   flfi_marker.Append(NULL); // NULL -> all attributes are active
}

void LinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                                      Array<int> &bdr_attr_marker)
{
   flfi.Append(lfi);
   flfi_marker.Append(&bdr_attr_marker);
}

void LinearForm::Assemble()
{
   Array<int> vdofs;
   ElementTransformation *eltrans;
   Vector elemvect;

   int i;

   Vector::operator=(0.0);

   if (dlfi.Size())
      for (i = 0; i < fes -> GetNE(); i++)
      {
         fes -> GetElementVDofs (i, vdofs);
         eltrans = fes -> GetElementTransformation (i);
         for (int k=0; k < dlfi.Size(); k++)
         {
            if (dlfi[k]->IsDelta()) { continue; }
            dlfi[k]->AssembleRHSElementVect(*fes->GetFE(i), *eltrans, elemvect);
            AddElementVector (vdofs, elemvect);
         }
      }

   AssembleDelta();

   if (blfi.Size())
      for (i = 0; i < fes -> GetNBE(); i++)
      {
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < blfi.Size(); k++)
         {
            blfi[k]->AssembleRHSElementVect(*fes->GetBE(i), *eltrans, elemvect);
            AddElementVector (vdofs, elemvect);
         }
      }

   if (flfi.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < flfi.Size(); k++)
      {
         if (flfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *flfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary face integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (i = 0; i < mesh->GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            for (int k = 0; k < flfi.Size(); k++)
            {
               if (flfi_marker[k] &&
                   (*flfi_marker[k])[bdr_attr-1] == 0) { continue; }

               flfi[k] -> AssembleRHSElementVect (*fes->GetFE(tr -> Elem1No),
                                                  *tr, elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }
}

void LinearForm::Update(FiniteElementSpace *f, Vector &v, int v_offset)
{
   fes = f;
   NewDataAndSize((double *)v + v_offset, fes->GetVSize());
   check_delta = true;
}

void LinearForm::AssembleDelta()
{
   if (check_delta)
   {
      int nc = 0;
      int sdim = fes->GetMesh()->SpaceDimension();
      dlfi_delta_elem_id.SetSize(dlfi.Size());
      dlfi_delta_ip.SetSize(dlfi.Size());
      dlfi_delta_lfid.SetSize(dlfi.Size());

      Array<double> acenters;
      acenters.Reserve(3*dlfi.Size());
      for (int i = 0; i < dlfi.Size(); i++)
      {
         Vector center;
         dlfi[i]->GetDeltaCenter(center);
         if (center.Size() > 0)
         {
            MFEM_VERIFY(center.Size() == sdim,
                        "Point dim " << center.Size() <<
                        " does not match space dim " << sdim)
            dlfi_delta_lfid[nc] = i;
            dlfi_delta_elem_id[nc] = -1;
            for (int k = 0; k < center.Size(); k++) { acenters.Append(center[k]); }
            nc++;
         }
      }
      Vector centers(acenters,nc*sdim);
      fes->GetMesh()->MatchPointsWithElemId(nc,centers,dlfi_delta_elem_id,
                                            dlfi_delta_ip);
      dlfi_delta_elem_id.SetSize(nc);
   }

   Array<int> vdofs;
   Vector elemvect;
   for (int i = 0; i < dlfi_delta_elem_id.Size(); i++)
   {
      int elem_id = dlfi_delta_elem_id[i];
      if (elem_id < 0) { continue; }
      int lfid = dlfi_delta_lfid[i];

      // define fake integration rule for point evaluation
      const IntegrationRule *save = dlfi[lfid]->GetIntRule();
      const IntegrationPoint &ip = dlfi_delta_ip[i];
      IntegrationRule fir(1);
      fir[0].Set(ip.x,ip.y,ip.z,1.0);
      dlfi[lfid]->SetIntRule(&fir);

      // define fake ElementTransformation
      FakeTransformation feltrans(fes->GetFE(elem_id)->GetDim());

      // pointwise evaluation
      fes -> GetElementVDofs (elem_id, vdofs);
      dlfi[lfid]->AllowDeltaEval();
      dlfi[lfid]->AssembleRHSElementVect(*fes->GetFE(elem_id), feltrans, elemvect);
      AddElementVector (vdofs, elemvect);
      dlfi[lfid]->AllowDeltaEval(false);
      dlfi[lfid]->SetIntRule(save);
   }
   check_delta = false;
}

LinearForm::~LinearForm()
{
   int k;
   for (k=0; k < dlfi.Size(); k++) { delete dlfi[k]; }
   for (k=0; k < blfi.Size(); k++) { delete blfi[k]; }
   for (k=0; k < flfi.Size(); k++) { delete flfi[k]; }
}

}
