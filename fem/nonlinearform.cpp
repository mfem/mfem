// Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "fem.hpp"
#include "../general/forall.hpp"

namespace mfem
{

NonlinearForm::NonlinearForm(NonlinearForm &&other)
   : Operator(other.fes->GetTrueVSize()), assembly(other.assembly),
     ext(other.ext), fes(other.fes), Grad(other.Grad), cGrad(other.cGrad),
     sequence(other.fes->GetSequence()), P(other.fes->GetProlongationMatrix()),
     cP(dynamic_cast<const SparseMatrix*>(P))
{
   // We swap stored integrators and markers with the moved nonlinear form
   mfem::Swap(domain_integs, other.domain_integs);
   mfem::Swap(domain_integs_marker, other.domain_integs_marker);
   mfem::Swap(interior_face_integs, other.interior_face_integs);
   mfem::Swap(boundary_face_integs, other.boundary_face_integs);
   mfem::Swap(boundary_face_integs_marker, other.boundary_face_integs_marker);

   /// Leave the moved nonlinear form in a state as if it was just constructed
   /// with fes
   other.ext = nullptr;
   other.cGrad = nullptr;
   other.Grad = nullptr;
   other.assembly = AssemblyLevel::LEGACY;
}

NonlinearForm& NonlinearForm::operator=(NonlinearForm &&other)
{
   if (this != &other)
   {
      /// Cleanup current nonlinear form first
      delete cGrad;
      delete Grad;
      for (int i = 0; i < domain_integs.Size(); i++) { delete domain_integs[i]; }
      for (int i = 0; i < interior_face_integs.Size(); i++) { delete interior_face_integs[i]; }
      for (int i = 0; i < boundary_face_integs.Size(); i++) { delete boundary_face_integs[i]; }
      delete ext;

      /// Null out all our integs and set size of their arrays to zero
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         domain_integs[k] = nullptr;
      }
      domain_integs.SetSize(0);
      for (int k = 0; k < boundary_face_integs.Size(); k++)
      {
         boundary_face_integs[k] = nullptr;
      }
      boundary_face_integs.SetSize(0);
      for (int k = 0; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k] = nullptr;
      }
      interior_face_integs.SetSize(0);

      /// Null out all our markers and set size of their arrays to zero
      for (int k = 0; k < domain_integs_marker.Size(); ++k)
      {
         domain_integs_marker[k] = nullptr;
      }
      domain_integs_marker.SetSize(0);
      for (int k = 0; k < boundary_face_integs_marker.Size(); ++k)
      {
         boundary_face_integs_marker[k] = nullptr;
      }
      boundary_face_integs_marker.SetSize(0);

      /// Now steal data from other nonlinear form leaving it in a state as if
      /// it was just constructed with fes
      Operator::operator=(std::move(other));

      assembly = other.assembly;
      other.assembly = AssemblyLevel::LEGACY;
      Grad = other.Grad;
      other.Grad = nullptr;
      cGrad = other.cGrad;
      other.cGrad = nullptr;

      // Swap our empty integ and marker arrays with the moved nonlinear form
      mfem::Swap(domain_integs, other.domain_integs);
      mfem::Swap(domain_integs_marker, other.domain_integs_marker);
      mfem::Swap(interior_face_integs, other.interior_face_integs);
      mfem::Swap(boundary_face_integs, other.boundary_face_integs);
      mfem::Swap(boundary_face_integs_marker, other.boundary_face_integs_marker);

      ext = other.ext;
      other.ext = nullptr;
   }
   return *this;
}

void NonlinearForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   if (ext)
   {
      MFEM_ABORT("the assembly level has already been set!");
   }
   assembly = assembly_level;
   switch (assembly)
   {
      case AssemblyLevel::NONE:
         ext = new MFNonlinearFormExtension(this);
         break;
      case AssemblyLevel::PARTIAL:
         ext = new PANonlinearFormExtension(this);
         break;
      case AssemblyLevel::LEGACY:
         // This is the default
         break;
      default:
         mfem_error("Unknown assembly level for this form.");
   }
}

void NonlinearForm::AddDomainIntegrator(NonlinearFormIntegrator *nlfi)
{
   domain_integs.Append(nlfi);
   domain_integs_marker.Append(nullptr); // null marker means apply everywhere
}

void NonlinearForm::AddDomainIntegrator(NonlinearFormIntegrator *nlfi,
                                        Array<int> &elem_marker)
{
   domain_integs.Append(nlfi);
   domain_integs_marker.Append(&elem_marker);
}

void NonlinearForm::AddInteriorFaceIntegrator(NonlinearFormIntegrator *nlfi)
{
   interior_face_integs.Append(nlfi);
}

void NonlinearForm::AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi)
{
   boundary_face_integs.Append(nlfi);
   // null marker means apply everywhere
   boundary_face_integs_marker.Append(nullptr);
}

void NonlinearForm::AddBdrFaceIntegrator(NonlinearFormIntegrator *nlfi,
                                         Array<int> &bdr_marker)
{
   boundary_face_integs.Append(nlfi);
   boundary_face_integs_marker.Append(&bdr_marker);
}

void NonlinearForm::SetEssentialBC(const Array<int> &bdr_attr_is_ess,
                                   Vector *rhs)
{
   // virtual call, works in parallel too
   fes->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list);

   if (rhs)
   {
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         (*rhs)(ess_tdof_list[i]) = 0.0;
      }
   }
}

void NonlinearForm::SetEssentialVDofs(const Array<int> &ess_vdofs_list)
{
   if (!P)
   {
      ess_vdofs_list.Copy(ess_tdof_list); // ess_vdofs_list --> ess_tdof_list
   }
   else
   {
      Array<int> ess_vdof_marker, ess_tdof_marker;
      FiniteElementSpace::ListToMarker(ess_vdofs_list, fes->GetVSize(),
                                       ess_vdof_marker);
      if (Serial())
      {
         fes->ConvertToConformingVDofs(ess_vdof_marker, ess_tdof_marker);
      }
      else
      {
#ifdef MFEM_USE_MPI
         ParFiniteElementSpace *pf = dynamic_cast<ParFiniteElementSpace*>(fes);
         ess_tdof_marker.SetSize(pf->GetTrueVSize());
         pf->Dof_TrueDof_Matrix()->BooleanMultTranspose(1, ess_vdof_marker,
                                                        0, ess_tdof_marker);
#else
         MFEM_ABORT("internal MFEM error");
#endif
      }
      FiniteElementSpace::MarkerToList(ess_tdof_marker, ess_tdof_list);
   }
}

double NonlinearForm::GetGridFunctionEnergy(const Vector &x) const
{
   if (ext)
   {
      MFEM_VERIFY(!interior_face_integs.Size(),
                  "Interior faces terms not yet implemented!");
      MFEM_VERIFY(!boundary_face_integs.Size(),
                  "Boundary face terms not yet implemented!");
      return ext->GetGridFunctionEnergy(x);
   }

   Array<int> vdofs;
   Vector el_x;
   const FiniteElement *fe;
   ElementTransformation *T;
   DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();

   double energy = 0.0;

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != nullptr)
         {
            MFEM_VERIFY(mesh->attributes.Size() ==
                        domain_integs_marker[k]->Size(),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }
      }

      for (int i = 0; i < fes->GetNE(); i++)
      {
         int elem_attr = mesh->GetAttribute(i);
         fe = fes->GetFE(i);
         doftrans = fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         x.GetSubVector(vdofs, el_x);
         if (doftrans) {doftrans->InvTransformPrimal(el_x); }
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if (domain_integs_marker[k] == nullptr ||
                (*(domain_integs_marker[k]))[elem_attr-1] == 1)
            {
               energy += domain_integs[k]->GetElementEnergy(*fe, *T, el_x);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);
            x.GetSubVector(vdofs, el_x);
            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);
            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               energy += interior_face_integs[k]->GetFaceEnergy(*fe1, *fe2, *tr, el_x);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

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

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            x.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               energy += boundary_face_integs[k]->GetFaceEnergy(*fe1, *fe2, *tr, el_x);
            }
         }
      }
   }
   return energy;
}

const Vector &NonlinearForm::Prolongate(const Vector &x) const
{
   MFEM_VERIFY(x.Size() == Width(), "invalid input Vector size");
   if (P)
   {
      aux1.SetSize(P->Height());
      P->Mult(x, aux1);
      return aux1;
   }
   return x;
}

void NonlinearForm::Mult(const Vector &x, Vector &y) const
{
   const Vector &px = Prolongate(x);
   if (P) { aux2.SetSize(P->Height()); }

   // If we are in parallel, ParNonLinearForm::Mult uses the aux2 vector. In
   // serial, place the result directly in y (when there is no P).
   Vector &py = P ? aux2 : y;

   if (ext)
   {
      ext->Mult(px, py);
      if (Serial())
      {
         if (cP) { cP->MultTranspose(py, y); }
         const int N = ess_tdof_list.Size();
         const auto tdof = ess_tdof_list.Read();
         auto Y = y.ReadWrite();
         MFEM_FORALL(i, N, Y[tdof[i]] = 0.0; );
      }
      // In parallel, the result is in 'py' which is an alias for 'aux2'.
      return;
   }

   Array<int> vdofs;
   Vector el_x, el_y;
   const FiniteElement *fe;
   ElementTransformation *T;
   DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();

   py = 0.0;

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != nullptr)
         {
            MFEM_VERIFY(mesh->attributes.Size() ==
                        domain_integs_marker[k]->Size(),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }
      }

      for (int i = 0; i < fes->GetNE(); i++)
      {
         int elem_attr = mesh->GetAttribute(i);
         fe = fes->GetFE(i);
         doftrans = fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         if (doftrans) {doftrans->InvTransformPrimal(el_x); }
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if (domain_integs_marker[k] == nullptr ||
                (*(domain_integs_marker[k]))[elem_attr-1] == 1)
            {
               domain_integs[k]->AssembleElementVector(*fe, *T, el_x, el_y);
               if (doftrans) {doftrans->TransformDual(el_y); }
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

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

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[k]->AssembleFaceVector(*fe1, *fe2, *tr, el_x, el_y);
               py.AddElementVector(vdofs, el_y);
            }
         }
      }
   }

   if (Serial())
   {
      if (cP) { cP->MultTranspose(py, y); }

      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         y(ess_tdof_list[i]) = 0.0;
      }
      // y(ess_tdof_list[i]) = x(ess_tdof_list[i]);
   }
   // In parallel, the result is in 'py' which is an alias for 'aux2'.
}

Operator &NonlinearForm::GetGradient(const Vector &x) const
{
   if (ext)
   {
      hGrad.Clear();
      Operator &grad = ext->GetGradient(Prolongate(x));
      Operator *Gop;
      grad.FormSystemOperator(ess_tdof_list, Gop);
      hGrad.Reset(Gop);
      // In both serial and parallel, when using extension, we return the final
      // global true-dof gradient with imposed b.c.
      return *hGrad;
   }

   const int skip_zeros = 0;
   Array<int> vdofs;
   Vector el_x;
   DenseMatrix elmat;
   const FiniteElement *fe;
   ElementTransformation *T;
   DofTransformation *doftrans;
   Mesh *mesh = fes->GetMesh();
   const Vector &px = Prolongate(x);

   if (Grad == NULL)
   {
      Grad = new SparseMatrix(fes->GetVSize());
   }
   else
   {
      *Grad = 0.0;
   }

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != nullptr)
         {
            MFEM_VERIFY(mesh->attributes.Size() ==
                        domain_integs_marker[k]->Size(),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }
      }

      for (int i = 0; i < fes->GetNE(); i++)
      {
         int elem_attr = fes->GetMesh()->GetAttribute(i);
         fe = fes->GetFE(i);
         doftrans = fes->GetElementVDofs(i, vdofs);
         T = fes->GetElementTransformation(i);
         px.GetSubVector(vdofs, el_x);
         if (doftrans) {doftrans->InvTransformPrimal(el_x); }
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if (domain_integs_marker[k] == nullptr ||
                (*(domain_integs_marker[k]))[elem_attr-1] == 1)
            {
               domain_integs[k]->AssembleElementGrad(*fe, *T, el_x, elmat);
               if (doftrans) { doftrans->TransformDual(elmat); }
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
               // Grad->AddSubMatrix(vdofs, vdofs, elmat, 1);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;
      Array<int> vdofs2;

      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append (vdofs2);

            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            fe2 = fes->GetFE(tr->Elem2No);

            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

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

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            px.GetSubVector(vdofs, el_x);

            fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[k]->AssembleFaceGrad(*fe1, *fe2, *tr, el_x, elmat);
               Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            }
         }
      }
   }

   if (!Grad->Finalized())
   {
      Grad->Finalize(skip_zeros);
   }

   SparseMatrix *mGrad = Grad;
   if (Serial())
   {
      if (cP)
      {
         delete cGrad;
         cGrad = RAP(*cP, *Grad, *cP);
         mGrad = cGrad;
      }
      for (int i = 0; i < ess_tdof_list.Size(); i++)
      {
         mGrad->EliminateRowCol(ess_tdof_list[i]);
      }
   }

   return *mGrad;
}

void NonlinearForm::Update()
{
   if (sequence == fes->GetSequence()) { return; }

   height = width = fes->GetTrueVSize();
   delete cGrad; cGrad = NULL;
   delete Grad; Grad = NULL;
   hGrad.Clear();
   ess_tdof_list.SetSize(0); // essential b.c. will need to be set again
   sequence = fes->GetSequence();
   // Do not modify aux1 and aux2, their size will be set before use.
   P = fes->GetProlongationMatrix();
   cP = dynamic_cast<const SparseMatrix*>(P);

   if (ext) { ext->Update(); }
}

void NonlinearForm::Setup()
{
   if (ext) { ext->Assemble(); }
}

NonlinearForm::~NonlinearForm()
{
   delete cGrad;
   delete Grad;
   for (int i = 0; i < domain_integs.Size(); i++) { delete domain_integs[i]; }
   for (int i = 0; i < interior_face_integs.Size(); i++) { delete interior_face_integs[i]; }
   for (int i = 0; i < boundary_face_integs.Size(); i++) { delete boundary_face_integs[i]; }
   delete ext;
}


BlockNonlinearForm::BlockNonlinearForm() :
   fes(0), BlockGrad(NULL)
{
   height = 0;
   width = 0;
}

void BlockNonlinearForm::SetSpaces(Array<FiniteElementSpace *> &f)
{
   delete BlockGrad;
   BlockGrad = NULL;
   for (int i=0; i<Grads.NumRows(); ++i)
   {
      for (int j=0; j<Grads.NumCols(); ++j)
      {
         delete Grads(i,j);
         delete cGrads(i,j);
      }
   }
   for (int i = 0; i < ess_tdofs.Size(); ++i)
   {
      delete ess_tdofs[i];
   }

   height = 0;
   width = 0;
   f.Copy(fes);
   block_offsets.SetSize(f.Size() + 1);
   block_trueOffsets.SetSize(f.Size() + 1);
   block_offsets[0] = 0;
   block_trueOffsets[0] = 0;

   for (int i=0; i<fes.Size(); ++i)
   {
      block_offsets[i+1] = fes[i]->GetVSize();
      block_trueOffsets[i+1] = fes[i]->GetTrueVSize();
   }

   block_offsets.PartialSum();
   block_trueOffsets.PartialSum();

   height = block_trueOffsets[fes.Size()];
   width = block_trueOffsets[fes.Size()];

   Grads.SetSize(fes.Size(), fes.Size());
   Grads = NULL;

   cGrads.SetSize(fes.Size(), fes.Size());
   cGrads = NULL;

   P.SetSize(fes.Size());
   cP.SetSize(fes.Size());
   ess_tdofs.SetSize(fes.Size());
   for (int s = 0; s < fes.Size(); ++s)
   {
      // Retrieve prolongation matrix for each FE space
      P[s] = fes[s]->GetProlongationMatrix();
      cP[s] = dynamic_cast<const SparseMatrix *>(P[s]);

      // If the P Operator exists and its type is not SparseMatrix, this
      // indicates the Operator is part of parallel run.
      if (P[s] && !cP[s])
      {
         is_serial = false;
      }

      // If the P Operator exists and its type is SparseMatrix, this indicates
      // the Operator is serial but needs prolongation on assembly.
      if (cP[s])
      {
         needs_prolongation = true;
      }

      ess_tdofs[s] = new Array<int>;
   }
}

BlockNonlinearForm::BlockNonlinearForm(Array<FiniteElementSpace *> &f) :
   fes(0), BlockGrad(NULL)
{
   SetSpaces(f);
}

void BlockNonlinearForm::AddBdrFaceIntegrator(BlockNonlinearFormIntegrator *nfi,
                                              Array<int> &bdr_attr_marker)
{
   boundary_face_integs.Append(nfi);
   boundary_face_integs_marker.Append(&bdr_attr_marker);
}

void BlockNonlinearForm::SetEssentialBC(
   const Array<Array<int> *> &bdr_attr_is_ess, Array<Vector *> &rhs)
{
   for (int s = 0; s < fes.Size(); ++s)
   {
      ess_tdofs[s]->SetSize(ess_tdofs.Size());

      fes[s]->GetEssentialTrueDofs(*bdr_attr_is_ess[s], *ess_tdofs[s]);

      if (rhs[s])
      {
         rhs[s]->SetSubVector(*ess_tdofs[s], 0.0);
      }
   }
}

double BlockNonlinearForm::GetEnergyBlocked(const BlockVector &bx) const
{
   Array<Array<int> *> vdofs(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array<const Vector *> el_x_const(fes.Size());
   Array<const FiniteElement *> fe(fes.Size());
   ElementTransformation *T;
   DofTransformation *doftrans;
   double energy = 0.0;

   for (int i=0; i<fes.Size(); ++i)
   {
      el_x_const[i] = el_x[i] = new Vector();
      vdofs[i] = new Array<int>;
   }

   if (domain_integs.Size())
      for (int i = 0; i < fes[0]->GetNE(); ++i)
      {
         T = fes[0]->GetElementTransformation(i);
         for (int s=0; s<fes.Size(); ++s)
         {
            fe[s] = fes[s]->GetFE(i);
            doftrans = fes[s]->GetElementVDofs(i, *vdofs[s]);
            bx.GetBlock(s).GetSubVector(*vdofs[s], *el_x[s]);
            if (doftrans) {doftrans->InvTransformPrimal(*el_x[s]); }
         }

         for (int k = 0; k < domain_integs.Size(); ++k)
         {
            energy += domain_integs[k]->GetElementEnergy(fe, *T, el_x_const);
         }
      }

   // free the allocated memory
   for (int i = 0; i < fes.Size(); ++i)
   {
      delete el_x[i];
      delete vdofs[i];
   }

   if (interior_face_integs.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from interior face terms");
   }

   if (boundary_face_integs.Size())
   {
      MFEM_ABORT("TODO: add energy contribution from boundary face terms");
   }

   return energy;
}

double BlockNonlinearForm::GetEnergy(const Vector &x) const
{
   xs.Update(const_cast<Vector&>(x), block_offsets);
   return GetEnergyBlocked(xs);
}

void BlockNonlinearForm::MultBlocked(const BlockVector &bx,
                                     BlockVector &by) const
{
   Array<Array<int> *>vdofs(fes.Size());
   Array<Array<int> *>vdofs2(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array<const Vector *> el_x_const(fes.Size());
   Array<Vector *> el_y(fes.Size());
   Array<const FiniteElement *> fe(fes.Size());
   Array<const FiniteElement *> fe2(fes.Size());
   ElementTransformation *T;
   Array<DofTransformation *> doftrans(fes.Size()); doftrans = nullptr;

   by.UseDevice(true);
   by = 0.0;
   by.SyncToBlocks();
   for (int s=0; s<fes.Size(); ++s)
   {
      el_x_const[s] = el_x[s] = new Vector();
      el_y[s] = new Vector();
      vdofs[s] = new Array<int>;
      vdofs2[s] = new Array<int>;
   }

   if (domain_integs.Size())
   {
      for (int i = 0; i < fes[0]->GetNE(); ++i)
      {
         T = fes[0]->GetElementTransformation(i);
         for (int s = 0; s < fes.Size(); ++s)
         {
            doftrans[s] = fes[s]->GetElementVDofs(i, *(vdofs[s]));
            fe[s] = fes[s]->GetFE(i);
            bx.GetBlock(s).GetSubVector(*(vdofs[s]), *el_x[s]);
            if (doftrans[s]) {doftrans[s]->InvTransformPrimal(*el_x[s]); }
         }

         for (int k = 0; k < domain_integs.Size(); ++k)
         {
            domain_integs[k]->AssembleElementVector(fe, *T,
                                                    el_x_const, el_y);

            for (int s=0; s<fes.Size(); ++s)
            {
               if (el_y[s]->Size() == 0) { continue; }
               if (doftrans[s]) {doftrans[s]->TransformDual(*el_y[s]); }
               by.GetBlock(s).AddElementVector(*(vdofs[s]), *el_y[s]);
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      Mesh *mesh = fes[0]->GetMesh();
      FaceElementTransformations *tr;

      for (int i = 0; i < mesh->GetNumFaces(); ++i)
      {
         tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            for (int s=0; s<fes.Size(); ++s)
            {
               fe[s] = fes[s]->GetFE(tr->Elem1No);
               fe2[s] = fes[s]->GetFE(tr->Elem2No);

               fes[s]->GetElementVDofs(tr->Elem1No, *(vdofs[s]));
               fes[s]->GetElementVDofs(tr->Elem2No, *(vdofs2[s]));

               vdofs[s]->Append(*(vdofs2[s]));

               bx.GetBlock(s).GetSubVector(*(vdofs[s]), *el_x[s]);
            }

            for (int k = 0; k < interior_face_integs.Size(); ++k)
            {

               interior_face_integs[k]->AssembleFaceVector(fe, fe2, *tr, el_x_const, el_y);

               for (int s=0; s<fes.Size(); ++s)
               {
                  if (el_y[s]->Size() == 0) { continue; }
                  by.GetBlock(s).AddElementVector(*(vdofs[s]), *el_y[s]);
               }
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      Mesh *mesh = fes[0]->GetMesh();
      FaceElementTransformations *tr;
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); ++k)
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
         for (int i = 0; i < bdr_attr_marker.Size(); ++i)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < mesh->GetNBE(); ++i)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            for (int s=0; s<fes.Size(); ++s)
            {
               fe[s] = fes[s]->GetFE(tr->Elem1No);
               fe2[s] = fes[s]->GetFE(tr->Elem1No);

               fes[s]->GetElementVDofs(tr->Elem1No, *(vdofs[s]));
               bx.GetBlock(s).GetSubVector(*(vdofs[s]), *el_x[s]);
            }

            for (int k = 0; k < boundary_face_integs.Size(); ++k)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[k]->AssembleFaceVector(fe, fe2, *tr, el_x_const, el_y);

               for (int s=0; s<fes.Size(); ++s)
               {
                  if (el_y[s]->Size() == 0) { continue; }
                  by.GetBlock(s).AddElementVector(*(vdofs[s]), *el_y[s]);
               }
            }
         }
      }
   }

   for (int s=0; s<fes.Size(); ++s)
   {
      delete vdofs2[s];
      delete vdofs[s];
      delete el_y[s];
      delete el_x[s];
   }

   by.SyncFromBlocks();
}

const BlockVector &BlockNonlinearForm::Prolongate(const BlockVector &bx) const
{
   MFEM_VERIFY(bx.Size() == Width(), "invalid input BlockVector size");

   if (needs_prolongation)
   {
      aux1.Update(block_offsets);
      for (int s = 0; s < fes.Size(); s++)
      {
         P[s]->Mult(bx.GetBlock(s), aux1.GetBlock(s));
      }
      return aux1;
   }
   return bx;
}

void BlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   BlockVector bx(const_cast<Vector&>(x), block_trueOffsets);
   BlockVector by(y, block_trueOffsets);

   const BlockVector &pbx = Prolongate(bx);
   if (needs_prolongation)
   {
      aux2.Update(block_offsets);
   }
   BlockVector &pby = needs_prolongation ? aux2 : by;

   xs.Update(const_cast<BlockVector&>(pbx), block_offsets);
   ys.Update(pby, block_offsets);
   MultBlocked(xs, ys);

   for (int s = 0; s < fes.Size(); s++)
   {
      if (cP[s])
      {
         cP[s]->MultTranspose(pby.GetBlock(s), by.GetBlock(s));
      }
      by.GetBlock(s).SetSubVector(*ess_tdofs[s], 0.0);
   }
}

void BlockNonlinearForm::ComputeGradientBlocked(const BlockVector &bx) const
{
   const int skip_zeros = 0;
   Array<Array<int> *> vdofs(fes.Size());
   Array<Array<int> *> vdofs2(fes.Size());
   Array<Vector *> el_x(fes.Size());
   Array<const Vector *> el_x_const(fes.Size());
   Array2D<DenseMatrix *> elmats(fes.Size(), fes.Size());
   Array<const FiniteElement *>fe(fes.Size());
   Array<const FiniteElement *>fe2(fes.Size());
   ElementTransformation * T;
   Array<DofTransformation *> doftrans(fes.Size()); doftrans = nullptr;

   for (int i=0; i<fes.Size(); ++i)
   {
      el_x_const[i] = el_x[i] = new Vector();
      vdofs[i] = new Array<int>;
      vdofs2[i] = new Array<int>;
      for (int j=0; j<fes.Size(); ++j)
      {
         elmats(i,j) = new DenseMatrix();
      }
   }

   for (int i=0; i<fes.Size(); ++i)
   {
      for (int j=0; j<fes.Size(); ++j)
      {
         if (Grads(i,j) != NULL)
         {
            *Grads(i,j) = 0.0;
         }
         else
         {
            Grads(i,j) = new SparseMatrix(fes[i]->GetVSize(),
                                          fes[j]->GetVSize());
         }
      }
   }

   if (domain_integs.Size())
   {
      for (int i = 0; i < fes[0]->GetNE(); ++i)
      {
         T = fes[0]->GetElementTransformation(i);
         for (int s = 0; s < fes.Size(); ++s)
         {
            fe[s] = fes[s]->GetFE(i);
            doftrans[s] = fes[s]->GetElementVDofs(i, *vdofs[s]);
            bx.GetBlock(s).GetSubVector(*vdofs[s], *el_x[s]);
            if (doftrans[s]) {doftrans[s]->InvTransformPrimal(*el_x[s]); }
         }

         for (int k = 0; k < domain_integs.Size(); ++k)
         {
            domain_integs[k]->AssembleElementGrad(fe, *T, el_x_const, elmats);

            for (int j=0; j<fes.Size(); ++j)
            {
               for (int l=0; l<fes.Size(); ++l)
               {
                  if (elmats(j,l)->Height() == 0) { continue; }
                  if (doftrans[j] || doftrans[l])
                  {
                     TransformDual(doftrans[j], doftrans[l], *elmats(j,l));
                  }
                  Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l],
                                           *elmats(j,l), skip_zeros);
               }
            }
         }
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes[0]->GetMesh();

      for (int i = 0; i < mesh->GetNumFaces(); ++i)
      {
         tr = mesh->GetInteriorFaceTransformations(i);

         for (int s=0; s < fes.Size(); ++s)
         {
            fe[s] = fes[s]->GetFE(tr->Elem1No);
            fe2[s] = fes[s]->GetFE(tr->Elem2No);

            fes[s]->GetElementVDofs(tr->Elem1No, *vdofs[s]);
            fes[s]->GetElementVDofs(tr->Elem2No, *vdofs2[s]);
            vdofs[s]->Append(*(vdofs2[s]));

            bx.GetBlock(s).GetSubVector(*vdofs[s], *el_x[s]);
         }

         for (int k = 0; k < interior_face_integs.Size(); ++k)
         {
            interior_face_integs[k]->AssembleFaceGrad(fe, fe2, *tr, el_x_const, elmats);
            for (int j=0; j<fes.Size(); ++j)
            {
               for (int l=0; l<fes.Size(); ++l)
               {
                  if (elmats(j,l)->Height() == 0) { continue; }
                  Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l],
                                           *elmats(j,l), skip_zeros);
               }
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Mesh *mesh = fes[0]->GetMesh();

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_face_integs.Size(); ++k)
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
         for (int i = 0; i < bdr_attr_marker.Size(); ++i)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < mesh->GetNBE(); ++i)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            for (int s = 0; s < fes.Size(); ++s)
            {
               fe[s] = fes[s]->GetFE(tr->Elem1No);
               fe2[s] = fe[s];

               fes[s]->GetElementVDofs(tr->Elem1No, *vdofs[s]);
               bx.GetBlock(s).GetSubVector(*vdofs[s], *el_x[s]);
            }

            for (int k = 0; k < boundary_face_integs.Size(); ++k)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }
               boundary_face_integs[k]->AssembleFaceGrad(fe, fe2, *tr, el_x_const, elmats);
               for (int l=0; l<fes.Size(); ++l)
               {
                  for (int j=0; j<fes.Size(); ++j)
                  {
                     if (elmats(j,l)->Height() == 0) { continue; }
                     Grads(j,l)->AddSubMatrix(*vdofs[j], *vdofs[l],
                                              *elmats(j,l), skip_zeros);
                  }
               }
            }
         }
      }
   }

   if (!Grads(0,0)->Finalized())
   {
      for (int i=0; i<fes.Size(); ++i)
      {
         for (int j=0; j<fes.Size(); ++j)
         {
            Grads(i,j)->Finalize(skip_zeros);
         }
      }
   }

   for (int i=0; i<fes.Size(); ++i)
   {
      for (int j=0; j<fes.Size(); ++j)
      {
         delete elmats(i,j);
      }
      delete vdofs2[i];
      delete vdofs[i];
      delete el_x[i];
   }
}

Operator &BlockNonlinearForm::GetGradient(const Vector &x) const
{
   BlockVector bx(const_cast<Vector&>(x), block_trueOffsets);
   const BlockVector &pbx = Prolongate(bx);

   ComputeGradientBlocked(pbx);

   Array2D<SparseMatrix *> mGrads(fes.Size(), fes.Size());
   mGrads = Grads;
   if (needs_prolongation)
   {
      for (int s1 = 0; s1 < fes.Size(); ++s1)
      {
         for (int s2 = 0; s2 < fes.Size(); ++s2)
         {
            delete cGrads(s1, s2);
            cGrads(s1, s2) = RAP(*cP[s1], *Grads(s1, s2), *cP[s2]);
            mGrads(s1, s2) = cGrads(s1, s2);
         }
      }
   }

   for (int s = 0; s < fes.Size(); ++s)
   {
      for (int i = 0; i < ess_tdofs[s]->Size(); ++i)
      {
         for (int j = 0; j < fes.Size(); ++j)
         {
            if (s == j)
            {
               mGrads(s, s)->EliminateRowCol((*ess_tdofs[s])[i],
                                             Matrix::DIAG_ONE);
            }
            else
            {
               mGrads(s, j)->EliminateRow((*ess_tdofs[s])[i]);
               mGrads(j, s)->EliminateCol((*ess_tdofs[s])[i]);
            }
         }
      }
   }

   delete BlockGrad;
   BlockGrad = new BlockOperator(block_trueOffsets);
   for (int i = 0; i < fes.Size(); ++i)
   {
      for (int j = 0; j < fes.Size(); ++j)
      {
         BlockGrad->SetBlock(i, j, mGrads(i, j));
      }
   }
   return *BlockGrad;
}

BlockNonlinearForm::~BlockNonlinearForm()
{
   delete BlockGrad;
   for (int i=0; i<fes.Size(); ++i)
   {
      for (int j=0; j<fes.Size(); ++j)
      {
         delete Grads(i,j);
         delete cGrads(i,j);
      }
      delete ess_tdofs[i];
   }

   for (int i = 0; i < domain_integs.Size(); ++i)
   {
      delete domain_integs[i];
   }

   for (int i = 0; i < interior_face_integs.Size(); ++i)
   {
      delete interior_face_integs[i];
   }

   for (int i = 0; i < boundary_face_integs.Size(); ++i)
   {
      delete boundary_face_integs[i];
   }

}

}
