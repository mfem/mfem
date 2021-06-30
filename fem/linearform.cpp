// Copyright (c) 2010-2021, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

// Implementation of class LinearForm

#include "fem.hpp"

namespace mfem
{

LinearForm::LinearForm(FiniteElementSpace *f, LinearForm *lf)
   : Vector(f->GetVSize())
{
   // Linear forms are stored on the device
   UseDevice(true);

   fes = f;
   extern_lfs = 1;

   // Copy the pointers to the integrators
   domain_integs = lf->domain_integs;

   domain_delta_integs = lf->domain_delta_integs;

   boundary_integs = lf->boundary_integs;

   boundary_face_integs = lf->boundary_face_integs;
   boundary_face_integs_marker = lf->boundary_face_integs_marker;
}

void LinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi)
{
   DeltaLFIntegrator *maybe_delta =
      dynamic_cast<DeltaLFIntegrator *>(lfi);
   if (!maybe_delta || !maybe_delta->IsDelta())
   {
      domain_integs.Append(lfi);
   }
   else
   {
      domain_delta_integs.Append(maybe_delta);
   }
   domain_integs_marker.Append(NULL);
}

void LinearForm::AddDomainIntegrator(LinearFormIntegrator *lfi,
                                     Array<int> &elem_marker)
{
   DeltaLFIntegrator *maybe_delta =
      dynamic_cast<DeltaLFIntegrator *>(lfi);
   if (!maybe_delta || !maybe_delta->IsDelta())
   {
      domain_integs.Append(lfi);
   }
   else
   {
      domain_delta_integs.Append(maybe_delta);
   }
   domain_integs_marker.Append(&elem_marker);
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi)
{
   boundary_integs.Append (lfi);
   boundary_integs_marker.Append(NULL); // NULL -> all attributes are active
}

void LinearForm::AddBoundaryIntegrator (LinearFormIntegrator * lfi,
                                        Array<int> &bdr_attr_marker)
{
   boundary_integs.Append (lfi);
   boundary_integs_marker.Append(&bdr_attr_marker);
}

void LinearForm::AddBdrFaceIntegrator (LinearFormIntegrator * lfi)
{
   boundary_face_integs.Append(lfi);
   // NULL -> all attributes are active
   boundary_face_integs_marker.Append(NULL);
}

void LinearForm::AddBdrFaceIntegrator(LinearFormIntegrator *lfi,
                                      Array<int> &bdr_attr_marker)
{
   boundary_face_integs.Append(lfi);
   boundary_face_integs_marker.Append(&bdr_attr_marker);
}

void LinearForm::AddInteriorFaceIntegrator(LinearFormIntegrator *lfi)
{
   interior_face_integs.Append(lfi);
}

void LinearForm::Assemble()
{
   Array<int> vdofs;
   ElementTransformation *eltrans;
   Vector elemvect;

   int i;

   Vector::operator=(0.0);

   // The above operation is executed on device because of UseDevice().
   // The first use of AddElementVector() below will move it back to host
   // because both 'vdofs' and 'elemvect' are on host.

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(fes->GetMesh()->attributes.Size() ==
                        domain_integs_marker[k]->Size(),
                        "invalid element marker for domain linear form "
                        "integrator #" << k << ", counting from zero");
         }
      }

      for (i = 0; i < fes -> GetNE(); i++)
      {
         int elem_attr = fes->GetMesh()->GetAttribute(i);
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if ( domain_integs_marker[k] == NULL ||
                 (*(domain_integs_marker[k]))[elem_attr-1] == 1 )
            {
               fes -> GetElementVDofs (i, vdofs);
               eltrans = fes -> GetElementTransformation (i);
               domain_integs[k]->AssembleRHSElementVect(*fes->GetFE(i),
                                                        *eltrans, elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }
   AssembleDelta();

   if (boundary_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_integs.Size(); k++)
      {
         if (boundary_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary integrator #"
                     << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         for (int k=0; k < boundary_integs.Size(); k++)
         {
            if (boundary_integs_marker[k] &&
                (*boundary_integs_marker[k])[bdr_attr-1] == 0) { continue; }

            boundary_integs[k]->AssembleRHSElementVect(*fes->GetBE(i),
                                                       *eltrans, elemvect);

            AddElementVector (vdofs, elemvect);
         }
      }
   }
   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         if (boundary_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_face_integs_marker[k];
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
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[k]->AssembleRHSElementVect(*fes->GetFE(tr->Elem1No),
                                                               *tr, elemvect);
               AddElementVector (vdofs, elemvect);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();

      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         for (i = 0; i < mesh->GetNumFaces(); i++)
         {
            FaceElementTransformations *tr = NULL;
            tr = mesh->GetInteriorFaceTransformations (i);
            if (tr != NULL)
            {
               fes -> GetElementVDofs (tr -> Elem1No, vdofs);
               Array<int> vdofs2;
               fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
               vdofs.Append(vdofs2);
               interior_face_integs[k]->AssembleRHSElementVect(*fes->GetFE(tr->Elem1No),
                                                               *fes->GetFE(tr -> Elem2No),
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
   NewMemoryAndSize(Memory<double>(v.GetMemory(), v_offset, f->GetVSize()),
                    f->GetVSize(), false);
   ResetDeltaLocations();
}

void LinearForm::MakeRef(FiniteElementSpace *f, Vector &v, int v_offset)
{
   MFEM_ASSERT(v.Size() >= v_offset + f->GetVSize(), "");
   fes = f;
   v.UseDevice(true);
   this->Vector::MakeRef(v, v_offset, fes->GetVSize());
}

void LinearForm::AssembleDelta()
{
   if (domain_delta_integs.Size() == 0) { return; }

   if (!HaveDeltaLocations())
   {
      int sdim = fes->GetMesh()->SpaceDimension();
      Vector center;
      DenseMatrix centers(sdim, domain_delta_integs.Size());
      for (int i = 0; i < centers.Width(); i++)
      {
         centers.GetColumnReference(i, center);
         domain_delta_integs[i]->GetDeltaCenter(center);
         MFEM_VERIFY(center.Size() == sdim,
                     "Point dim " << center.Size() <<
                     " does not match space dim " << sdim);
      }
      fes->GetMesh()->FindPoints(centers, domain_delta_integs_elem_id,
                                 domain_delta_integs_ip);
   }

   Array<int> vdofs;
   Vector elemvect;
   for (int i = 0; i < domain_delta_integs.Size(); i++)
   {
      int elem_id = domain_delta_integs_elem_id[i];
      // The delta center may be outside of this sub-domain, or
      // (Par)Mesh::FindPoints() failed to find this point:
      if (elem_id < 0) { continue; }

      const IntegrationPoint &ip = domain_delta_integs_ip[i];
      ElementTransformation &Trans = *fes->GetElementTransformation(elem_id);
      Trans.SetIntPoint(&ip);

      fes->GetElementVDofs(elem_id, vdofs);
      domain_delta_integs[i]->AssembleDeltaElementVect(*fes->GetFE(elem_id),
                                                       Trans, elemvect);
      AddElementVector(vdofs, elemvect);
   }
}

LinearForm & LinearForm::operator=(double value)
{
   Vector::operator=(value);
   return *this;
}

LinearForm & LinearForm::operator=(const Vector &v)
{
   MFEM_ASSERT(fes && v.Size() == fes->GetVSize(), "");
   Vector::operator=(v);
   return *this;
}

LinearForm::~LinearForm()
{
   if (!extern_lfs)
   {
      int k;
      for (k=0; k < domain_delta_integs.Size(); k++)
      { delete domain_delta_integs[k]; }
      for (k=0; k < domain_integs.Size(); k++) { delete domain_integs[k]; }
      for (k=0; k < boundary_integs.Size(); k++) { delete boundary_integs[k]; }
      for (k=0; k < boundary_face_integs.Size(); k++)
      { delete boundary_face_integs[k]; }
      for (k=0; k < interior_face_integs.Size(); k++)
      { delete interior_face_integs[k]; }
   }
}

}
