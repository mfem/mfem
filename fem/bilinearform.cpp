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

// Implementation of class BilinearForm

#include "fem.hpp"
#include "../general/device.hpp"
#include "../mesh/nurbs.hpp"
#include <cmath>

namespace mfem
{

void BilinearForm::AllocMat()
{
   if (static_cond) { return; }

   if (precompute_sparsity == 0 || fes->GetVDim() > 1)
   {
      mat = new SparseMatrix(height);
      return;
   }

   const Table &elem_dof = fes->GetElementToDofTable();
   Table dof_dof;

   if (interior_face_integs.Size() > 0)
   {
      // the sparsity pattern is defined from the map: face->element->dof
      Table face_dof, dof_face;
      {
         Table *face_elem = fes->GetMesh()->GetFaceToElementTable();
         mfem::Mult(*face_elem, elem_dof, face_dof);
         delete face_elem;
      }
      Transpose(face_dof, dof_face, height);
      mfem::Mult(dof_face, face_dof, dof_dof);
   }
   else
   {
      // the sparsity pattern is defined from the map: element->dof
      Table dof_elem;
      Transpose(elem_dof, dof_elem, height);
      mfem::Mult(dof_elem, elem_dof, dof_dof);
   }

   dof_dof.SortRows();

   int *I = dof_dof.GetI();
   int *J = dof_dof.GetJ();
   real_t *data = Memory<real_t>(I[height]);

   mat = new SparseMatrix(I, J, data, height, height, true, true, true);
   *mat = 0.0;

   dof_dof.LoseData();
}

BilinearForm::BilinearForm(FiniteElementSpace * f)
   : Matrix (f->GetVSize())
{
   fes = f;
   sequence = f->GetSequence();
   mat = mat_e = NULL;
   extern_bfs = 0;
   element_matrices = NULL;
   static_cond = NULL;
   hybridization = NULL;
   precompute_sparsity = 0;
   diag_policy = DIAG_KEEP;

   assembly = AssemblyLevel::LEGACY;
   batch = 1;
   ext = NULL;
}

BilinearForm::BilinearForm (FiniteElementSpace * f, BilinearForm * bf, int ps)
   : Matrix (f->GetVSize())
{
   fes = f;
   sequence = f->GetSequence();
   mat_e = NULL;
   extern_bfs = 1;
   element_matrices = NULL;
   static_cond = NULL;
   hybridization = NULL;
   precompute_sparsity = ps;
   diag_policy = DIAG_KEEP;

   assembly = AssemblyLevel::LEGACY;
   batch = 1;
   ext = NULL;

   // Copy the pointers to the integrators
   domain_integs = bf->domain_integs;
   domain_integs_marker = bf->domain_integs_marker;

   boundary_integs = bf->boundary_integs;
   boundary_integs_marker = bf->boundary_integs_marker;

   interior_face_integs = bf->interior_face_integs;

   boundary_face_integs = bf->boundary_face_integs;
   boundary_face_integs_marker = bf->boundary_face_integs_marker;

   AllocMat();
}

void BilinearForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   if (ext)
   {
      MFEM_ABORT("the assembly level has already been set!");
   }
   assembly = assembly_level;
   switch (assembly)
   {
      case AssemblyLevel::LEGACY:
         break;
      case AssemblyLevel::FULL:
         SetDiagonalPolicy( DIAG_ONE ); // Only diagonal policy supported on device
         ext = new FABilinearFormExtension(this);
         break;
      case AssemblyLevel::ELEMENT:
         ext = new EABilinearFormExtension(this);
         break;
      case AssemblyLevel::PARTIAL:
         ext = new PABilinearFormExtension(this);
         break;
      case AssemblyLevel::NONE:
         ext = new MFBilinearFormExtension(this);
         break;
      default:
         MFEM_ABORT("BilinearForm: unknown assembly level");
   }
}

void BilinearForm::EnableStaticCondensation()
{
   delete static_cond;
   if (assembly != AssemblyLevel::LEGACY)
   {
      static_cond = NULL;
      MFEM_WARNING("Static condensation not supported for this assembly level");
      return;
   }
   static_cond = new StaticCondensation(fes);
   if (static_cond->ReducesTrueVSize())
   {
      bool symmetric = false;      // TODO
      bool block_diagonal = false; // TODO
      static_cond->Init(symmetric, block_diagonal);
   }
   else
   {
      delete static_cond;
      static_cond = NULL;
   }
}

void BilinearForm::EnableHybridization(FiniteElementSpace *constr_space,
                                       BilinearFormIntegrator *constr_integ,
                                       const Array<int> &ess_tdof_list)
{
   delete hybridization;
   if (assembly != AssemblyLevel::LEGACY)
   {
      delete constr_integ;
      hybridization = NULL;
      MFEM_WARNING("Hybridization not supported for this assembly level");
      return;
   }
   hybridization = new Hybridization(fes, constr_space);
   hybridization->SetConstraintIntegrator(constr_integ);
   hybridization->Init(ess_tdof_list);
}

void BilinearForm::UseSparsity(int *I, int *J, bool isSorted)
{
   if (static_cond) { return; }

   if (mat)
   {
      if (mat->Finalized() && mat->GetI() == I && mat->GetJ() == J)
      {
         return; // mat is already using the given sparsity
      }
      delete mat;
   }
   height = width = fes->GetVSize();
   mat = new SparseMatrix(I, J, NULL, height, width, false, true, isSorted);
}

void BilinearForm::UseSparsity(SparseMatrix &A)
{
   MFEM_ASSERT(A.Height() == fes->GetVSize() && A.Width() == fes->GetVSize(),
               "invalid matrix A dimensions: "
               << A.Height() << " x " << A.Width());
   MFEM_ASSERT(A.Finalized(), "matrix A must be Finalized");

   UseSparsity(A.GetI(), A.GetJ(), A.ColumnsAreSorted());
}

real_t& BilinearForm::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const real_t& BilinearForm::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

MatrixInverse * BilinearForm::Inverse() const
{
   return mat -> Inverse();
}

void BilinearForm::Finalize (int skip_zeros)
{
   if (assembly == AssemblyLevel::LEGACY)
   {
      if (!static_cond) { mat->Finalize(skip_zeros); }
      if (mat_e) { mat_e->Finalize(skip_zeros); }
      if (static_cond) { static_cond->Finalize(); }
      if (hybridization) { hybridization->Finalize(); }
   }
}

void BilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi)
{
   domain_integs.Append(bfi);
   domain_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void BilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi,
                                       Array<int> &elem_marker)
{
   domain_integs.Append(bfi);
   domain_integs_marker.Append(&elem_marker);
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   boundary_integs.Append (bfi);
   boundary_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi,
                                          Array<int> &bdr_marker)
{
   boundary_integs.Append (bfi);
   boundary_integs_marker.Append(&bdr_marker);
}

void BilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator * bfi)
{
   interior_face_integs.Append (bfi);
}

void BilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi)
{
   boundary_face_integs.Append(bfi);
   // NULL marker means apply everywhere
   boundary_face_integs_marker.Append(NULL);
}

void BilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
                                        Array<int> &bdr_marker)
{
   boundary_face_integs.Append(bfi);
   boundary_face_integs_marker.Append(&bdr_marker);
}

void BilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat) const
{
   if (element_matrices)
   {
      elmat.SetSize(element_matrices->SizeI(), element_matrices->SizeJ());
      elmat = element_matrices->GetData(i);
      return;
   }

   const FiniteElement &fe = *fes->GetFE(i);

   if (domain_integs.Size())
   {
      ElementTransformation *eltrans = fes->GetElementTransformation(i);
      domain_integs[0]->AssembleElementMatrix(fe, *eltrans, elmat);
      for (int k = 1; k < domain_integs.Size(); k++)
      {
         domain_integs[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int ndof = fe.GetDof() * fes->GetVDim();
      elmat.SetSize(ndof);
      elmat = 0.0;
   }
}

void BilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat) const
{
   const FiniteElement &be = *fes->GetBE(i);

   if (boundary_integs.Size())
   {
      ElementTransformation *eltrans = fes->GetBdrElementTransformation(i);
      boundary_integs[0]->AssembleElementMatrix(be, *eltrans, elmat);
      for (int k = 1; k < boundary_integs.Size(); k++)
      {
         boundary_integs[k]->AssembleElementMatrix(be, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int ndof = be.GetDof() * fes->GetVDim();
      elmat.SetSize(ndof);
      elmat = 0.0;
   }
}

void BilinearForm::ComputeFaceMatrix(int i, DenseMatrix &elmat) const
{
   FaceElementTransformations *tr;
   Mesh *mesh = fes -> GetMesh();
   tr = mesh -> GetFaceElementTransformations (i);

   const FiniteElement *fe1, *fe2;
   fe1 = fes->GetFE(tr->Elem1No);
   if (tr->Elem2No >= 0)
   {
      fe2 = fes->GetFE(tr->Elem2No);
   }
   else
   {
      // The fe2 object is really a dummy and not used on the
      // boundaries, but we can't dereference a NULL pointer, and we don't
      // want to actually make a fake element.
      fe2 = fe1;
   }

   if (interior_face_integs.Size())
   {
      interior_face_integs[0] -> AssembleFaceMatrix (*fe1, *fe2, *tr, elmat);
      for (int k = 1; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k] -> AssembleFaceMatrix (*fe1, *fe2, *tr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      int ndof = fe1->GetDof() * fes->GetVDim();
      if (tr->Elem2No >= 0)
      {
         ndof += fe2->GetDof() * fes->GetVDim();
      }

      elmat.SetSize(ndof);
      elmat = 0.0;
   }
}

void BilinearForm::ComputeBdrFaceMatrix(int i, DenseMatrix &elmat) const
{
   FaceElementTransformations *tr;
   Mesh *mesh = fes -> GetMesh();
   tr = mesh -> GetBdrFaceTransformations (i);

   const FiniteElement *fe1, *fe2;

   fe1 = fes -> GetFE (tr -> Elem1No);
   // The fe2 object is really a dummy and not used on the boundaries,
   // but we can't dereference a NULL pointer, and we don't want to
   // actually make a fake element.
   fe2 = fe1;

   if (boundary_face_integs.Size())
   {
      boundary_face_integs[0] -> AssembleFaceMatrix (*fe1, *fe2, *tr, elmat);
      for (int k = 1; k < boundary_face_integs.Size(); k++)
      {
         boundary_face_integs[k] -> AssembleFaceMatrix (*fe1, *fe2, *tr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      int ndof = fe1->GetDof() * fes->GetVDim();
      elmat.SetSize(ndof);
      elmat = 0.0;
   }
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleElementMatrix(i, elmat, vdofs, skip_zeros);
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &vdofs_, int skip_zeros)
{
   fes->GetElementVDofs(i, vdofs_);
   if (static_cond)
   {
      static_cond->AssembleMatrix(i, elmat);
   }
   else
   {
      if (mat == NULL)
      {
         AllocMat();
      }
      mat->AddSubMatrix(vdofs_, vdofs_, elmat, skip_zeros);
      if (hybridization)
      {
         hybridization->AssembleMatrix(i, elmat);
      }
   }
}

void BilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleBdrElementMatrix(i, elmat, vdofs, skip_zeros);
}

void BilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &vdofs_, int skip_zeros)
{
   fes->GetBdrElementVDofs(i, vdofs_);
   if (static_cond)
   {
      static_cond->AssembleBdrMatrix(i, elmat);
   }
   else
   {
      if (mat == NULL)
      {
         AllocMat();
      }
      mat->AddSubMatrix(vdofs_, vdofs_, elmat, skip_zeros);
      if (hybridization)
      {
         hybridization->AssembleBdrMatrix(i, elmat);
      }
   }
}

void BilinearForm::Assemble(int skip_zeros)
{
   if (ext)
   {
      ext->Assemble();
      return;
   }

   ElementTransformation *eltrans;
   DofTransformation * doftrans;
   Mesh *mesh = fes -> GetMesh();
   DenseMatrix elmat, *elmat_p;

   if (mat == NULL)
   {
      AllocMat();
   }

#ifdef MFEM_USE_LEGACY_OPENMP
   int free_element_matrices = 0;
   if (!element_matrices)
   {
      ComputeElementMatrices();
      free_element_matrices = 1;
   }
#endif

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }

         if (domain_integs[k]->Patchwise())
         {
            MFEM_VERIFY(fes->GetNURBSext(), "Patchwise integration requires a "
                        << "NURBS FE space");
         }
      }

      // Element-wise integration
      for (int i = 0; i < fes -> GetNE(); i++)
      {
         // Set both doftrans (potentially needed to assemble the element
         // matrix) and vdofs, which is also needed when the element matrices
         // are pre-assembled.
         doftrans = fes->GetElementVDofs(i, vdofs);
         if (element_matrices)
         {
            elmat_p = &(*element_matrices)(i);
         }
         else
         {
            const int elem_attr = fes->GetMesh()->GetAttribute(i);
            eltrans = fes->GetElementTransformation(i);

            elmat.SetSize(0);
            for (int k = 0; k < domain_integs.Size(); k++)
            {
               if ((domain_integs_marker[k] == NULL ||
                    (*(domain_integs_marker[k]))[elem_attr-1] == 1)
                   && !domain_integs[k]->Patchwise())
               {
                  domain_integs[k]->AssembleElementMatrix(*fes->GetFE(i),
                                                          *eltrans, elemmat);
                  if (elmat.Size() == 0)
                  {
                     elmat = elemmat;
                  }
                  else
                  {
                     elmat += elemmat;
                  }
               }
            }
            if (elmat.Size() == 0)
            {
               continue;
            }
            else
            {
               elmat_p = &elmat;
            }
            if (doftrans)
            {
               doftrans->TransformDual(elmat);
            }
            elmat_p = &elmat;
         }
         if (static_cond)
         {
            static_cond->AssembleMatrix(i, *elmat_p);
         }
         else
         {
            mat->AddSubMatrix(vdofs, vdofs, *elmat_p, skip_zeros);
            if (hybridization)
            {
               hybridization->AssembleMatrix(i, *elmat_p);
            }
         }
      }

      // Patch-wise integration
      if (fes->GetNURBSext())
      {
         for (int p=0; p<mesh->NURBSext->GetNP(); ++p)
         {
            bool vdofsSet = false;
            for (int k = 0; k < domain_integs.Size(); k++)
            {
               if (domain_integs[k]->Patchwise())
               {
                  if (!vdofsSet)
                  {
                     fes->GetPatchVDofs(p, vdofs);
                     vdofsSet = true;
                  }

                  SparseMatrix* spmat = nullptr;
                  domain_integs[k]->AssemblePatchMatrix(p, *fes, spmat);
                  Array<int> cols;
                  Vector srow;

                  for (int r=0; r<spmat->Height(); ++r)
                  {
                     spmat->GetRow(r, cols, srow);
                     for (int i=0; i<cols.Size(); ++i)
                     {
                        cols[i] = vdofs[cols[i]];
                     }
                     mat->AddRow(vdofs[r], cols, srow);
                  }

                  delete spmat;
               }
            }
         }
      }
   }

   if (boundary_integs.Size())
   {
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

      for (int i = 0; i < fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         const FiniteElement &be = *fes->GetBE(i);
         doftrans = fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         int k = 0;
         for (; k < boundary_integs.Size(); k++)
         {
            if (boundary_integs_marker[k] &&
                (*boundary_integs_marker[k])[bdr_attr-1] == 0) { continue; }

            boundary_integs[k]->AssembleElementMatrix(be, *eltrans, elmat);
            k++;
            break;
         }
         for (; k < boundary_integs.Size(); k++)
         {
            if (boundary_integs_marker[k] &&
                (*boundary_integs_marker[k])[bdr_attr-1] == 0) { continue; }

            boundary_integs[k]->AssembleElementMatrix(be, *eltrans, elemmat);
            elmat += elemmat;
         }
         if (doftrans)
         {
            doftrans->TransformDual(elmat);
         }
         elmat_p = &elmat;
         if (!static_cond)
         {
            mat->AddSubMatrix(vdofs, vdofs, *elmat_p, skip_zeros);
            if (hybridization)
            {
               hybridization->AssembleBdrMatrix(i, *elmat_p);
            }
         }
         else
         {
            static_cond->AssembleBdrMatrix(i, *elmat_p);
         }
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *tr;
      Array<int> vdofs2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         tr = mesh -> GetInteriorFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            fes -> GetElementVDofs (tr -> Elem2No, vdofs2);
            vdofs.Append (vdofs2);
            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->
               AssembleFaceMatrix(*fes->GetFE(tr->Elem1No),
                                  *fes->GetFE(tr->Elem2No),
                                  *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
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

         tr = mesh -> GetBdrFaceTransformations (i);
         if (tr != NULL)
         {
            fes -> GetElementVDofs (tr -> Elem1No, vdofs);
            fe1 = fes -> GetFE (tr -> Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            fe2 = fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0)
               { continue; }

               boundary_face_integs[k] -> AssembleFaceMatrix (*fe1, *fe2, *tr,
                                                              elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

#ifdef MFEM_USE_LEGACY_OPENMP
   if (free_element_matrices)
   {
      FreeElementMatrices();
   }
#endif
}

void BilinearForm::ConformingAssemble()
{
   // Do not remove zero entries to preserve the symmetric structure of the
   // matrix which in turn will give rise to symmetric structure in the new
   // matrix. This ensures that subsequent calls to EliminateRowCol will work
   // correctly.
   Finalize(0);
   MFEM_ASSERT(mat, "the BilinearForm is not assembled");

   const SparseMatrix *P = fes->GetConformingProlongation();
   if (!P) { return; } // conforming mesh

   SparseMatrix *R = Transpose(*P);
   SparseMatrix *RA = mfem::Mult(*R, *mat);
   delete mat;
   if (mat_e)
   {
      SparseMatrix *RAe = mfem::Mult(*R, *mat_e);
      delete mat_e;
      mat_e = RAe;
   }
   delete R;
   mat = mfem::Mult(*RA, *P);
   delete RA;
   if (mat_e)
   {
      SparseMatrix *RAeP = mfem::Mult(*mat_e, *P);
      delete mat_e;
      mat_e = RAeP;
   }

   height = mat->Height();
   width = mat->Width();
}

void BilinearForm::AssembleDiagonal(Vector &diag) const
{
   MFEM_ASSERT(diag.Size() == fes->GetTrueVSize(),
               "Vector for holding diagonal has wrong size!");
   const SparseMatrix *cP = fes->GetConformingProlongation();
   if (!ext)
   {
      MFEM_ASSERT(mat, "the BilinearForm is not assembled!");
      MFEM_ASSERT(cP == nullptr || mat->Height() == cP->Width(),
                  "BilinearForm::ConformingAssemble() is not called!");
      mat->GetDiag(diag);
      return;
   }
   // Here, we have extension, ext.
   if (!cP)
   {
      ext->AssembleDiagonal(diag);
      return;
   }
   // Here, we have extension, ext, and conforming prolongation, cP.

   // For an AMR mesh, a convergent diagonal is assembled with |P^T| d_l,
   // where |P^T| has the entry-wise absolute values of the conforming
   // prolongation transpose operator.
   Vector local_diag(cP->Height());
   ext->AssembleDiagonal(local_diag);
   cP->AbsMultTranspose(local_diag, diag);
}

void BilinearForm::FormLinearSystem(const Array<int> &ess_tdof_list, Vector &x,
                                    Vector &b, OperatorHandle &A, Vector &X,
                                    Vector &B, int copy_interior)
{
   if (ext)
   {
      ext->FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);
      return;
   }
   const SparseMatrix *P = fes->GetConformingProlongation();
   FormSystemMatrix(ess_tdof_list, A);

   // Transform the system and perform the elimination in B, based on the
   // essential BC values from x. Restrict the BC part of x in X, and set the
   // non-BC part to zero. Since there is no good initial guess for the Lagrange
   // multipliers, set X = 0.0 for hybridization.
   if (static_cond)
   {
      // Schur complement reduction to the exposed dofs
      static_cond->ReduceSystem(x, b, X, B, copy_interior);
   }
   else if (!P) // conforming space
   {
      if (hybridization)
      {
         // Reduction to the Lagrange multipliers system
         EliminateVDofsInRHS(ess_tdof_list, x, b);
         hybridization->ReduceRHS(b, B);
         X.SetSize(B.Size());
         X = 0.0;
      }
      else
      {
         // A, X and B point to the same data as mat, x and b
         EliminateVDofsInRHS(ess_tdof_list, x, b);
         X.MakeRef(x, 0, x.Size());
         B.MakeRef(b, 0, b.Size());
         if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
      }
   }
   else // non-conforming space
   {
      if (hybridization)
      {
         // Reduction to the Lagrange multipliers system
         const SparseMatrix *R = fes->GetConformingRestriction();
         Vector conf_b(P->Width()), conf_x(P->Width());
         P->MultTranspose(b, conf_b);
         R->Mult(x, conf_x);
         EliminateVDofsInRHS(ess_tdof_list, conf_x, conf_b);
         R->MultTranspose(conf_b, b); // store eliminated rhs in b
         hybridization->ReduceRHS(conf_b, B);
         X.SetSize(B.Size());
         X = 0.0;
      }
      else
      {
         // Variational restriction with P
         const SparseMatrix *R = fes->GetConformingRestriction();
         B.SetSize(P->Width());
         P->MultTranspose(b, B);
         X.SetSize(R->Height());
         R->Mult(x, X);
         EliminateVDofsInRHS(ess_tdof_list, X, B);
         if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
      }
   }
}

void BilinearForm::FormSystemMatrix(const Array<int> &ess_tdof_list,
                                    OperatorHandle &A)
{
   if (ext)
   {
      ext->FormSystemMatrix(ess_tdof_list, A);
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   if (static_cond)
   {
      if (!static_cond->HasEliminatedBC())
      {
         static_cond->SetEssentialTrueDofs(ess_tdof_list);
         static_cond->Finalize(); // finalize Schur complement (to true dofs)
         static_cond->EliminateReducedTrueDofs(diag_policy);
         static_cond->Finalize(); // finalize eliminated part
      }
      A.Reset(&static_cond->GetMatrix(), false);
   }
   else
   {
      if (!mat_e)
      {
         const SparseMatrix *P = fes->GetConformingProlongation();
         if (P) { ConformingAssemble(); }
         EliminateVDofs(ess_tdof_list, diag_policy);
         const int remove_zeros = 0;
         Finalize(remove_zeros);
      }
      if (hybridization)
      {
         A.Reset(&hybridization->GetMatrix(), false);
      }
      else
      {
         A.Reset(mat, false);
      }
   }
}

void BilinearForm::RecoverFEMSolution(const Vector &X,
                                      const Vector &b, Vector &x)
{
   if (ext)
   {
      ext->RecoverFEMSolution(X, b, x);
      return;
   }

   const SparseMatrix *P = fes->GetConformingProlongation();
   if (!P) // conforming space
   {
      if (static_cond)
      {
         // Private dofs back solve
         static_cond->ComputeSolution(b, X, x);
      }
      else if (hybridization)
      {
         // Primal unknowns recovery
         hybridization->ComputeSolution(b, X, x);
      }
      else
      {
         // X and x point to the same data

         // If the validity flags of X's Memory were changed (e.g. if it was
         // moved to device memory) then we need to tell x about that.
         x.SyncMemory(X);
      }
   }
   else // non-conforming space
   {
      if (static_cond)
      {
         // Private dofs back solve
         static_cond->ComputeSolution(b, X, x);
      }
      else if (hybridization)
      {
         // Primal unknowns recovery
         Vector conf_b(P->Width()), conf_x(P->Width());
         P->MultTranspose(b, conf_b);
         const SparseMatrix *R = fes->GetConformingRestriction();
         R->Mult(x, conf_x); // get essential b.c. from x
         hybridization->ComputeSolution(conf_b, X, conf_x);
         x.SetSize(P->Height());
         P->Mult(conf_x, x);
      }
      else
      {
         // Apply conforming prolongation
         x.SetSize(P->Height());
         P->Mult(X, x);
      }
   }
}

void BilinearForm::ComputeElementMatrices()
{
   if (element_matrices || domain_integs.Size() == 0 || fes->GetNE() == 0)
   {
      return;
   }

   int num_elements = fes->GetNE();
   int num_dofs_per_el = fes->GetTypicalFE()->GetDof() * fes->GetVDim();

   element_matrices = new DenseTensor(num_dofs_per_el, num_dofs_per_el,
                                      num_elements);

   DenseMatrix tmp;
   IsoparametricTransformation eltrans;

#ifdef MFEM_USE_LEGACY_OPENMP
   #pragma omp parallel for private(tmp,eltrans)
#endif
   for (int i = 0; i < num_elements; i++)
   {
      DenseMatrix elmat(element_matrices->GetData(i),
                        num_dofs_per_el, num_dofs_per_el);
      const FiniteElement &fe = *fes->GetFE(i);
#ifdef MFEM_DEBUG
      if (num_dofs_per_el != fe.GetDof()*fes->GetVDim())
         mfem_error("BilinearForm::ComputeElementMatrices:"
                    " all elements must have same number of dofs");
#endif
      fes->GetElementTransformation(i, &eltrans);

      domain_integs[0]->AssembleElementMatrix(fe, eltrans, elmat);
      for (int k = 1; k < domain_integs.Size(); k++)
      {
         // note: some integrators may not be thread-safe
         domain_integs[k]->AssembleElementMatrix(fe, eltrans, tmp);
         elmat += tmp;
      }
      elmat.ClearExternalData();
   }
}

void BilinearForm::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                        const Vector &sol, Vector &rhs,
                                        DiagonalPolicy dpolicy)
{
   Array<int> ess_dofs, conf_ess_dofs;
   fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   if (fes->GetVSize() == height)
   {
      EliminateEssentialBCFromDofs(ess_dofs, sol, rhs, dpolicy);
   }
   else
   {
      fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
      EliminateEssentialBCFromDofs(conf_ess_dofs, sol, rhs, dpolicy);
   }
}

void BilinearForm::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
                                        DiagonalPolicy dpolicy)
{
   Array<int> ess_dofs, conf_ess_dofs;
   fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   if (fes->GetVSize() == height)
   {
      EliminateEssentialBCFromDofs(ess_dofs, dpolicy);
   }
   else
   {
      fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
      EliminateEssentialBCFromDofs(conf_ess_dofs, dpolicy);
   }
}

void BilinearForm::EliminateEssentialBCDiag (const Array<int> &bdr_attr_is_ess,
                                             real_t value)
{
   Array<int> ess_dofs, conf_ess_dofs;
   fes->GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

   if (fes->GetVSize() == height)
   {
      EliminateEssentialBCFromDofsDiag(ess_dofs, value);
   }
   else
   {
      fes->GetRestrictionMatrix()->BooleanMult(ess_dofs, conf_ess_dofs);
      EliminateEssentialBCFromDofsDiag(conf_ess_dofs, value);
   }
}

void BilinearForm::EliminateVDofs(const Array<int> &vdofs_,
                                  const Vector &sol, Vector &rhs,
                                  DiagonalPolicy dpolicy)
{
   vdofs_.HostRead();
   for (int i = 0; i < vdofs_.Size(); i++)
   {
      int vdof = vdofs_[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, sol(vdof), rhs, dpolicy);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, sol(-1-vdof), rhs, dpolicy);
      }
   }
}

void BilinearForm::EliminateVDofs(const Array<int> &vdofs_,
                                  DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(height);
   }

   vdofs_.HostRead();
   for (int i = 0; i < vdofs_.Size(); i++)
   {
      int vdof = vdofs_[i];
      if ( vdof >= 0 )
      {
         mat -> EliminateRowCol (vdof, *mat_e, dpolicy);
      }
      else
      {
         mat -> EliminateRowCol (-1-vdof, *mat_e, dpolicy);
      }
   }
}

void BilinearForm::EliminateEssentialBCFromDofs(
   const Array<int> &ess_dofs, const Vector &sol, Vector &rhs,
   DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");
   MFEM_ASSERT(sol.Size() == height, "incorrect sol Vector size");
   MFEM_ASSERT(rhs.Size() == height, "incorrect rhs Vector size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, sol(i), rhs, dpolicy);
      }
}

void BilinearForm::EliminateEssentialBCFromDofs (const Array<int> &ess_dofs,
                                                 DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(ess_dofs.Size() == height,
               "incorrect dof Array size: " << ess_dofs.Size() << ' ' << height);

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, dpolicy);
      }
}

void BilinearForm::EliminateEssentialBCFromDofsDiag (const Array<int> &ess_dofs,
                                                     real_t value)
{
   MFEM_ASSERT(ess_dofs.Size() == height,
               "incorrect dof Array size: " << ess_dofs.Size() << ' ' << height);

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowColDiag (i, value);
      }
}

void BilinearForm::EliminateVDofsInRHS(
   const Array<int> &vdofs_, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs_, x, b);
}

void BilinearForm::Mult(const Vector &x, Vector &y) const
{
   if (ext)
   {
      ext->Mult(x, y);
   }
   else
   {
      mat->Mult(x, y);
   }
}

void BilinearForm::MultTranspose(const Vector & x, Vector & y) const
{
   if (ext)
   {
      ext->MultTranspose(x, y);
   }
   else
   {
      y = 0.0;
      AddMultTranspose (x, y);
   }
}

void BilinearForm::Update(FiniteElementSpace *nfes)
{
   bool full_update;

   if (nfes && nfes != fes)
   {
      full_update = true;
      fes = nfes;
   }
   else
   {
      // Check for different size (e.g. assembled form on non-conforming space)
      // or different sequence number.
      full_update = (fes->GetVSize() != Height() ||
                     sequence < fes->GetSequence());
   }

   delete mat_e;
   mat_e = NULL;
   FreeElementMatrices();
   delete static_cond;
   static_cond = NULL;

   if (full_update)
   {
      delete mat;
      mat = NULL;
      delete hybridization;
      hybridization = NULL;
      sequence = fes->GetSequence();
   }
   else
   {
      if (mat) { *mat = 0.0; }
      if (hybridization) { hybridization->Reset(); }
   }

   height = width = fes->GetVSize();

   if (ext) { ext->Update(); }
}

void BilinearForm::SetDiagonalPolicy(DiagonalPolicy policy)
{
   diag_policy = policy;
}

BilinearForm::~BilinearForm()
{
   delete mat_e;
   delete mat;
   delete element_matrices;
   delete static_cond;
   delete hybridization;

   if (!extern_bfs)
   {
      int k;
      for (k=0; k < domain_integs.Size(); k++) { delete domain_integs[k]; }
      for (k=0; k < boundary_integs.Size(); k++) { delete boundary_integs[k]; }
      for (k=0; k < interior_face_integs.Size(); k++)
      { delete interior_face_integs[k]; }
      for (k=0; k < boundary_face_integs.Size(); k++)
      { delete boundary_face_integs[k]; }
   }

   delete ext;
}


MixedBilinearForm::MixedBilinearForm (FiniteElementSpace *tr_fes,
                                      FiniteElementSpace *te_fes)
   : Matrix(te_fes->GetVSize(), tr_fes->GetVSize())
{
   trial_fes = tr_fes;
   test_fes = te_fes;
   mat = NULL;
   mat_e = NULL;
   extern_bfs = 0;
   assembly = AssemblyLevel::LEGACY;
   ext = NULL;
}

MixedBilinearForm::MixedBilinearForm (FiniteElementSpace *tr_fes,
                                      FiniteElementSpace *te_fes,
                                      MixedBilinearForm * mbf)
   : Matrix(te_fes->GetVSize(), tr_fes->GetVSize())
{
   trial_fes = tr_fes;
   test_fes = te_fes;
   mat = NULL;
   mat_e = NULL;
   extern_bfs = 1;
   ext = NULL;

   // Copy the pointers to the integrators
   domain_integs = mbf->domain_integs;
   domain_integs_marker = mbf->domain_integs_marker;

   boundary_integs = mbf->boundary_integs;
   boundary_integs_marker = mbf->boundary_integs_marker;

   trace_face_integs = mbf->trace_face_integs;

   boundary_trace_face_integs = mbf->boundary_trace_face_integs;
   boundary_trace_face_integs_marker = mbf->boundary_trace_face_integs_marker;

   assembly = AssemblyLevel::LEGACY;
   ext = NULL;
}

void MixedBilinearForm::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   if (ext)
   {
      MFEM_ABORT("the assembly level has already been set!");
   }
   assembly = assembly_level;
   switch (assembly)
   {
      case AssemblyLevel::LEGACY:
         break;
      case AssemblyLevel::FULL:
         // ext = new FAMixedBilinearFormExtension(this);
         // Use the original BilinearForm implementation for now
         break;
      case AssemblyLevel::ELEMENT:
         mfem_error("Element assembly not supported yet... stay tuned!");
         // ext = new EAMixedBilinearFormExtension(this);
         break;
      case AssemblyLevel::PARTIAL:
         ext = new PAMixedBilinearFormExtension(this);
         break;
      case AssemblyLevel::NONE:
         mfem_error("Matrix-free action not supported yet... stay tuned!");
         // ext = new MFMixedBilinearFormExtension(this);
         break;
      default:
         mfem_error("Unknown assembly level");
   }
}

real_t & MixedBilinearForm::Elem (int i, int j)
{
   return (*mat)(i, j);
}

const real_t & MixedBilinearForm::Elem (int i, int j) const
{
   return (*mat)(i, j);
}

void MixedBilinearForm::Mult(const Vector & x, Vector & y) const
{
   y = 0.0;
   AddMult(x, y);
}

void MixedBilinearForm::AddMult(const Vector & x, Vector & y,
                                const real_t a) const
{
   if (ext)
   {
      ext->AddMult(x, y, a);
   }
   else
   {
      mat->AddMult(x, y, a);
   }
}

void MixedBilinearForm::MultTranspose(const Vector & x, Vector & y) const
{
   y = 0.0;
   AddMultTranspose(x, y);
}

void MixedBilinearForm::AddMultTranspose(const Vector & x, Vector & y,
                                         const real_t a) const
{
   if (ext)
   {
      ext->AddMultTranspose(x, y, a);
   }
   else
   {
      mat->AddMultTranspose(x, y, a);
   }
}

MatrixInverse * MixedBilinearForm::Inverse() const
{
   if (assembly != AssemblyLevel::LEGACY)
   {
      MFEM_WARNING("MixedBilinearForm::Inverse not possible with this "
                   "assembly level!");
      return NULL;
   }
   else
   {
      return mat -> Inverse ();
   }
}

void MixedBilinearForm::Finalize (int skip_zeros)
{
   if (assembly == AssemblyLevel::LEGACY)
   {
      mat -> Finalize (skip_zeros);
   }
}

void MixedBilinearForm::GetBlocks(Array2D<SparseMatrix *> &blocks) const
{
   MFEM_VERIFY(trial_fes->GetOrdering() == Ordering::byNODES &&
               test_fes->GetOrdering() == Ordering::byNODES,
               "MixedBilinearForm::GetBlocks: both trial and test spaces "
               "must use Ordering::byNODES!");

   blocks.SetSize(test_fes->GetVDim(), trial_fes->GetVDim());

   mat->GetBlocks(blocks);
}

void MixedBilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi)
{
   domain_integs.Append(bfi);
   domain_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void MixedBilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi,
                                            Array<int> &elem_marker)
{
   domain_integs.Append(bfi);
   domain_integs_marker.Append(&elem_marker);
}

void MixedBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi)
{
   boundary_integs.Append(bfi);
   boundary_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void MixedBilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi,
                                              Array<int> &bdr_marker)
{
   boundary_integs.Append(bfi);
   boundary_integs_marker.Append(&bdr_marker);
}

void MixedBilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi)
{
   interior_face_integs.Append(bfi);
}

void MixedBilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi)
{
   boundary_face_integs.Append(bfi);
   boundary_face_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void MixedBilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
                                             Array<int> &bdr_marker)
{
   boundary_face_integs.Append(bfi);
   boundary_face_integs_marker.Append(&bdr_marker);
}

void MixedBilinearForm::AddTraceFaceIntegrator (BilinearFormIntegrator * bfi)
{
   trace_face_integs.Append (bfi);
}

void MixedBilinearForm::AddBdrTraceFaceIntegrator(BilinearFormIntegrator *bfi)
{
   boundary_trace_face_integs.Append(bfi);
   // NULL marker means apply everywhere
   boundary_trace_face_integs_marker.Append(NULL);
}

void MixedBilinearForm::AddBdrTraceFaceIntegrator(BilinearFormIntegrator *bfi,
                                                  Array<int> &bdr_marker)
{
   boundary_trace_face_integs.Append(bfi);
   boundary_trace_face_integs_marker.Append(&bdr_marker);
}

void MixedBilinearForm::Assemble(int skip_zeros)
{
   if (ext)
   {
      ext->Assemble();
      return;
   }

   ElementTransformation *eltrans;
   DofTransformation * dom_dof_trans;
   DofTransformation * ran_dof_trans;
   DenseMatrix elmat;

   Mesh *mesh = test_fes -> GetMesh();

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }
      }

      for (int i = 0; i < test_fes -> GetNE(); i++)
      {
         const int elem_attr = mesh->GetAttribute(i);
         dom_dof_trans = trial_fes -> GetElementVDofs (i, trial_vdofs);
         ran_dof_trans = test_fes  -> GetElementVDofs (i, test_vdofs);
         eltrans = test_fes -> GetElementTransformation (i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if (domain_integs_marker[k] == NULL ||
                (*(domain_integs_marker[k]))[elem_attr-1] == 1)
            {
               domain_integs[k] -> AssembleElementMatrix2 (*trial_fes -> GetFE(i),
                                                           *test_fes  -> GetFE(i),
                                                           *eltrans, elemmat);
               elmat += elemmat;
            }
         }
         if (ran_dof_trans || dom_dof_trans)
         {
            TransformDual(ran_dof_trans, dom_dof_trans, elmat);
         }
         mat -> AddSubMatrix (test_vdofs, trial_vdofs, elmat, skip_zeros);
      }
   }

   if (boundary_integs.Size())
   {
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

      for (int i = 0; i < test_fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         dom_dof_trans = trial_fes -> GetBdrElementVDofs (i, trial_vdofs);
         ran_dof_trans = test_fes  -> GetBdrElementVDofs (i, test_vdofs);
         eltrans = test_fes -> GetBdrElementTransformation (i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int k = 0; k < boundary_integs.Size(); k++)
         {
            if (boundary_integs_marker[k] &&
                (*boundary_integs_marker[k])[bdr_attr-1] == 0) { continue; }

            boundary_integs[k]->AssembleElementMatrix2 (*trial_fes -> GetBE(i),
                                                        *test_fes  -> GetBE(i),
                                                        *eltrans, elemmat);
            elmat += elemmat;
         }
         if (ran_dof_trans || dom_dof_trans)
         {
            TransformDual(ran_dof_trans, dom_dof_trans, elmat);
         }
         mat -> AddSubMatrix (test_vdofs, trial_vdofs, elmat, skip_zeros);
      }
   }

   if (interior_face_integs.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> trial_vdofs2, test_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         ftr = mesh->GetInteriorFaceTransformations(i);
         if (ftr != NULL)
         {
            trial_fes->GetElementVDofs(ftr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
            trial_fe1 = trial_fes->GetFE(ftr->Elem1No);
            test_fe1 = test_fes->GetFE(ftr->Elem1No);
            if (ftr->Elem2No >= 0)
            {
               trial_fes->GetElementVDofs(ftr->Elem2No, trial_vdofs2);
               test_fes->GetElementVDofs(ftr->Elem2No, test_vdofs2);
               trial_vdofs.Append(trial_vdofs2);
               test_vdofs.Append(test_vdofs2);
               trial_fe2 = trial_fes->GetFE(ftr->Elem2No);
               test_fe2 = test_fes->GetFE(ftr->Elem2No);
            }
            else
            {
               // The test_fe2 object is really a dummy and not used on the
               // boundaries, but we can't dereference a NULL pointer, and we don't
               // want to actually make a fake element.
               trial_fe2 = trial_fe1;
               test_fe2 = test_fe1;
            }
            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                           *test_fe2,
                                                           *ftr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> tr_vdofs2, te_vdofs2;
      const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

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

      for (int i = 0; i < trial_fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         ftr = mesh -> GetBdrFaceTransformations (i);
         if (ftr != NULL)
         {
            trial_fes->GetElementVDofs(ftr->Elem1No, trial_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
            trial_fe1 = trial_fes->GetFE(ftr->Elem1No);
            test_fe1 = test_fes->GetFE(ftr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            trial_fe2 = trial_fe1;
            test_fe2 = test_fe1;
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] &&
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 0) { continue; }

               boundary_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                           *test_fe2,
                                                           *ftr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (trace_face_integs.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> test_vdofs2;
      const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         ftr = mesh->GetFaceElementTransformations(i);
         trial_fes->GetFaceVDofs(i, trial_vdofs);
         test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
         trial_face_fe = trial_fes->GetFaceElement(i);
         test_fe1 = test_fes->GetFE(ftr->Elem1No);
         if (ftr->Elem2No >= 0)
         {
            test_fes->GetElementVDofs(ftr->Elem2No, test_vdofs2);
            test_vdofs.Append(test_vdofs2);
            test_fe2 = test_fes->GetFE(ftr->Elem2No);
         }
         else
         {
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            test_fe2 = test_fe1;
         }
         for (int k = 0; k < trace_face_integs.Size(); k++)
         {
            trace_face_integs[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1,
                                                     *test_fe2, *ftr, elemmat);
            mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (boundary_trace_face_integs.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> te_vdofs2;
      const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < boundary_trace_face_integs.Size(); k++)
      {
         if (boundary_trace_face_integs_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *boundary_trace_face_integs_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary trace face"
                     "integrator #" << k << ", counting from zero");
         for (int i = 0; i < bdr_attr_marker.Size(); i++)
         {
            bdr_attr_marker[i] |= bdr_marker[i];
         }
      }

      for (int i = 0; i < trial_fes -> GetNBE(); i++)
      {
         const int bdr_attr = mesh->GetBdrAttribute(i);
         if (bdr_attr_marker[bdr_attr-1] == 0) { continue; }

         ftr = mesh->GetBdrFaceTransformations(i);
         if (ftr)
         {
            const int iface = mesh->GetBdrElementFaceIndex(i);
            trial_fes->GetFaceVDofs(iface, trial_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
            trial_face_fe = trial_fes->GetFaceElement(iface);
            test_fe1 = test_fes->GetFE(ftr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            test_fe2 = test_fe1;
            for (int k = 0; k < boundary_trace_face_integs.Size(); k++)
            {
               if (boundary_trace_face_integs_marker[k] &&
                   (*boundary_trace_face_integs_marker[k])[bdr_attr-1] == 0)
               { continue; }

               boundary_trace_face_integs[k]->AssembleFaceMatrix(*trial_face_fe,
                                                                 *test_fe1,
                                                                 *test_fe2,
                                                                 *ftr, elemmat);
               mat->AddSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }
}

void MixedBilinearForm::AssembleDiagonal_ADAt(const Vector &D,
                                              Vector &diag) const
{
   if (ext)
   {
      MFEM_ASSERT(diag.Size() == test_fes->GetTrueVSize(),
                  "Vector for holding diagonal has wrong size!");
      MFEM_ASSERT(D.Size() == trial_fes->GetTrueVSize(),
                  "Vector for holding diagonal has wrong size!");
      const Operator *P_trial = trial_fes->GetProlongationMatrix();
      const Operator *P_test = test_fes->GetProlongationMatrix();
      if (!IsIdentityProlongation(P_trial))
      {
         Vector local_D(P_trial->Height());
         P_trial->Mult(D, local_D);

         if (!IsIdentityProlongation(P_test))
         {
            Vector local_diag(P_test->Height());
            ext->AssembleDiagonal_ADAt(local_D, local_diag);
            P_test->MultTranspose(local_diag, diag);
         }
         else
         {
            ext->AssembleDiagonal_ADAt(local_D, diag);
         }
      }
      else
      {
         if (!IsIdentityProlongation(P_test))
         {
            Vector local_diag(P_test->Height());
            ext->AssembleDiagonal_ADAt(D, local_diag);
            P_test->MultTranspose(local_diag, diag);
         }
         else
         {
            ext->AssembleDiagonal_ADAt(D, diag);
         }
      }
   }
   else
   {
      MFEM_ABORT("Not implemented. Maybe assemble your bilinear form into a "
                 "matrix and use SparseMatrix functions?");
   }
}

void MixedBilinearForm::ConformingAssemble()
{
   if (assembly != AssemblyLevel::LEGACY)
   {
      MFEM_WARNING("Conforming assemble not supported for this assembly level!");
      return;
   }

   Finalize();

   const SparseMatrix *P2 = test_fes->GetConformingProlongation();
   if (P2)
   {
      SparseMatrix *R = Transpose(*P2);
      SparseMatrix *RA = mfem::Mult(*R, *mat);
      delete R;
      delete mat;
      mat = RA;
   }

   const SparseMatrix *P1 = trial_fes->GetConformingProlongation();
   if (P1)
   {
      SparseMatrix *RAP = mfem::Mult(*mat, *P1);
      delete mat;
      mat = RAP;
   }

   height = mat->Height();
   width = mat->Width();
}


void MixedBilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat) const
{
   const FiniteElement &trial_fe = *trial_fes->GetFE(i);
   const FiniteElement &test_fe = *test_fes->GetFE(i);

   if (domain_integs.Size())
   {
      ElementTransformation *eltrans = test_fes->GetElementTransformation(i);
      domain_integs[0]->AssembleElementMatrix2(trial_fe, test_fe, *eltrans,
                                               elmat);
      for (int k = 1; k < domain_integs.Size(); k++)
      {
         domain_integs[k]->AssembleElementMatrix2(trial_fe, test_fe, *eltrans,
                                                  elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int tr_dofs = trial_fe.GetDof() * trial_fes->GetVDim();
      const int te_dofs = test_fe.GetDof() * test_fes->GetVDim();

      elmat.SetSize(te_dofs, tr_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat) const
{
   const FiniteElement &trial_be = *trial_fes->GetBE(i);
   const FiniteElement &test_be = *test_fes->GetBE(i);

   if (boundary_integs.Size())
   {
      ElementTransformation *eltrans = test_fes->GetBdrElementTransformation(i);
      boundary_integs[0]->AssembleElementMatrix2(trial_be, test_be, *eltrans,
                                                 elmat);
      for (int k = 1; k < boundary_integs.Size(); k++)
      {
         boundary_integs[k]->AssembleElementMatrix2(trial_be, test_be, *eltrans,
                                                    elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int tr_dofs = trial_be.GetDof() * trial_fes->GetVDim();
      const int te_dofs = test_be.GetDof() * test_fes->GetVDim();

      elmat.SetSize(te_dofs, tr_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeFaceMatrix(int i, DenseMatrix &elmat) const
{
   FaceElementTransformations *ftr;
   Mesh *mesh = test_fes -> GetMesh();
   ftr = mesh->GetFaceElementTransformations(i);
   MFEM_ASSERT(ftr, "No associated face transformations.");

   const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

   trial_fe1 = trial_fes->GetFE(ftr->Elem1No);
   test_fe1 = test_fes->GetFE(ftr->Elem1No);
   if (ftr->Elem2No >= 0)
   {
      trial_fe2 = trial_fes->GetFE(ftr->Elem2No);
      test_fe2 = test_fes->GetFE(ftr->Elem2No);
   }
   else
   {
      // The test_fe2 object is really a dummy and not used on the
      // boundaries, but we can't dereference a NULL pointer, and we don't
      // want to actually make a fake element.
      trial_fe2 = trial_fe1;
      test_fe2 = test_fe1;
   }

   if (interior_face_integs.Size())
   {
      interior_face_integs[0]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                  *test_fe2,
                                                  *ftr, elmat);
      for (int k = 1; k < interior_face_integs.Size(); k++)
      {
         interior_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                     *test_fe2,
                                                     *ftr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      int tr_dofs = trial_fe1->GetDof() * trial_fes->GetVDim();
      int te_dofs = test_fe1->GetDof() * test_fes->GetVDim();
      if (ftr->Elem2No >= 0)
      {
         tr_dofs += trial_fe2->GetDof() * trial_fes->GetVDim();
         te_dofs += test_fe2->GetDof() * test_fes->GetVDim();
      }

      elmat.SetSize(te_dofs, tr_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeBdrFaceMatrix(int i, DenseMatrix &elmat) const
{
   FaceElementTransformations *ftr;
   Mesh *mesh = test_fes -> GetMesh();
   ftr = mesh->GetBdrFaceTransformations(i);
   MFEM_ASSERT(ftr, "No associated boundary face.");

   const FiniteElement *trial_fe1, *trial_fe2, *test_fe1, *test_fe2;

   trial_fe1 = trial_fes->GetFE(ftr->Elem1No);
   test_fe1 = test_fes->GetFE(ftr->Elem1No);
   // The test_fe2 object is really a dummy and not used on the
   // boundaries, but we can't dereference a NULL pointer, and we don't
   // want to actually make a fake element.
   trial_fe2 = trial_fe1;
   test_fe2 = test_fe1;

   if (boundary_face_integs.Size())
   {
      boundary_face_integs[0]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                  *test_fe2,
                                                  *ftr, elmat);
      for (int k = 1; k < boundary_face_integs.Size(); k++)
      {
         boundary_face_integs[k]->AssembleFaceMatrix(*trial_fe1, *test_fe1, *trial_fe2,
                                                     *test_fe2,
                                                     *ftr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int tr_dofs = trial_fe1->GetDof() * trial_fes->GetVDim();
      const int te_dofs = test_fe1->GetDof() * test_fes->GetVDim();

      elmat.SetSize(te_dofs, tr_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeTraceFaceMatrix(int i, DenseMatrix &elmat) const
{
   FaceElementTransformations *ftr;
   Mesh *mesh = test_fes -> GetMesh();
   ftr = mesh->GetFaceElementTransformations(i);
   MFEM_ASSERT(ftr, "No associated face transformation.");

   const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

   trial_face_fe = trial_fes->GetFaceElement(i);
   test_fe1 = test_fes->GetFE(ftr->Elem1No);
   if (ftr->Elem2No >= 0)
   {
      test_fe2 = test_fes->GetFE(ftr->Elem2No);
   }
   else
   {
      // The test_fe2 object is really a dummy and not used on the
      // boundaries, but we can't dereference a NULL pointer, and we don't
      // want to actually make a fake element.
      test_fe2 = test_fe1;
   }

   if (trace_face_integs.Size())
   {
      trace_face_integs[0]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
                                               *ftr, elmat);
      for (int k = 1; k < trace_face_integs.Size(); k++)
      {
         trace_face_integs[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
                                                  *ftr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int tr_face_dofs = trial_face_fe->GetDof() * trial_fes->GetVDim();
      int te_dofs = test_fe1->GetDof() * test_fes->GetVDim();
      if (ftr->Elem2No >= 0)
      {
         te_dofs += test_fe2->GetDof() * test_fes->GetVDim();
      }

      elmat.SetSize(te_dofs, tr_face_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeBdrTraceFaceMatrix(int i,
                                                  DenseMatrix &elmat) const
{
   FaceElementTransformations *ftr;
   Mesh *mesh = test_fes -> GetMesh();
   ftr = mesh->GetBdrFaceTransformations(i);
   MFEM_ASSERT(ftr, "No associated boundary face.");

   const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;
   int iface = mesh->GetBdrElementFaceIndex(i);
   trial_face_fe = trial_fes->GetFaceElement(iface);
   test_fe1 = test_fes->GetFE(ftr->Elem1No);
   // The test_fe2 object is really a dummy and not used on the
   // boundaries, but we can't dereference a NULL pointer, and we don't
   // want to actually make a fake element.
   test_fe2 = test_fe1;

   if (boundary_trace_face_integs.Size())
   {
      boundary_trace_face_integs[0]->AssembleFaceMatrix(*trial_face_fe, *test_fe1,
                                                        *test_fe2,
                                                        *ftr, elmat);
      for (int k = 1; k < boundary_trace_face_integs.Size(); k++)
      {
         boundary_trace_face_integs[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1,
                                                           *test_fe2,
                                                           *ftr, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      const int tr_face_dofs = trial_face_fe->GetDof() * trial_fes->GetVDim();
      int te_dofs = test_fe1->GetDof() * test_fes->GetVDim();

      elmat.SetSize(te_dofs, tr_face_dofs);
      elmat = 0.0;
   }
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &trial_vdofs_,
   Array<int> &test_vdofs_, int skip_zeros)
{
   trial_fes->GetElementVDofs(i, trial_vdofs_);
   test_fes->GetElementVDofs(i, test_vdofs_);
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs_, trial_vdofs_, elmat, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleBdrElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &trial_vdofs_,
   Array<int> &test_vdofs_, int skip_zeros)
{
   trial_fes->GetBdrElementVDofs(i, trial_vdofs_);
   test_fes->GetBdrElementVDofs(i, test_vdofs_);
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs_, trial_vdofs_, elmat, skip_zeros);
}

void MixedBilinearForm::EliminateTrialEssentialBC(
   const Array<int> &bdr_attr_is_ess, const Vector &sol, Vector &rhs )
{
   Array<int> trial_ess_dofs;
   trial_fes->GetEssentialVDofs(bdr_attr_is_ess, trial_ess_dofs);
   mat->EliminateCols(trial_ess_dofs, &sol, &rhs);
}

void MixedBilinearForm::EliminateTrialEssentialBC(const Array<int>
                                                  &bdr_attr_is_ess)
{
   Array<int> trial_ess_dofs;
   trial_fes->GetEssentialVDofs(bdr_attr_is_ess, trial_ess_dofs);
   mat->EliminateCols(trial_ess_dofs);
}

void MixedBilinearForm::EliminateTrialVDofs(const Array<int> &trial_vdofs_,
                                            const Vector &sol, Vector &rhs)
{
   Array<int> trial_vdofs_marker;
   FiniteElementSpace::ListToMarker(trial_vdofs_, mat->Width(),
                                    trial_vdofs_marker);
   mat->EliminateCols(trial_vdofs_marker, &sol, &rhs);
}

void MixedBilinearForm::EliminateTrialVDofs(const Array<int> &trial_vdofs_)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(mat->Height(), mat->Width());
   }

   Array<int> trial_vdofs_marker;
   FiniteElementSpace::ListToMarker(trial_vdofs_, mat->Width(),
                                    trial_vdofs_marker);
   mat->EliminateCols(trial_vdofs_marker, *mat_e);
   mat_e->Finalize();
}

void MixedBilinearForm::EliminateTrialVDofsInRHS(const Array<int> &trial_vdofs_,
                                                 const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
}

void MixedBilinearForm::EliminateEssentialBCFromTrialDofs(
   const Array<int> &marked_vdofs, const Vector &sol, Vector &rhs)
{
   mat->EliminateCols(marked_vdofs, &sol, &rhs);
}

void MixedBilinearForm::EliminateTestEssentialBC(const Array<int>
                                                 &bdr_attr_is_ess)
{
   int i, j, k;
   Array<int> te_vdofs;

   for (i = 0; i < test_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[test_fes -> GetBdrAttribute (i)-1])
      {
         test_fes -> GetBdrElementVDofs (i, te_vdofs);
         for (j = 0; j < te_vdofs.Size(); j++)
         {
            if ( (k = te_vdofs[j]) < 0 )
            {
               k = -1-k;
            }
            mat -> EliminateRow (k);
         }
      }
}

void MixedBilinearForm::EliminateTestVDofs(const Array<int> &test_vdofs_)
{
   for (int i=0; i<test_vdofs_.Size(); ++i)
   {
      mat->EliminateRow(test_vdofs_[i]);
   }
}

void MixedBilinearForm::FormRectangularSystemMatrix(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)

{
   if (ext)
   {
      ext->FormRectangularSystemOperator(trial_tdof_list, test_tdof_list, A);
      return;
   }

   const SparseMatrix *test_P = test_fes->GetConformingProlongation();
   const SparseMatrix *trial_P = trial_fes->GetConformingProlongation();

   mat->Finalize();

   if (test_P && trial_P)
   {
      SparseMatrix *m = RAP(*test_P, *mat, *trial_P);
      delete mat;
      mat = m;
   }
   else if (test_P)
   {
      SparseMatrix *m = TransposeMult(*test_P, *mat);
      delete mat;
      mat = m;
   }
   else if (trial_P)
   {
      SparseMatrix *m = mfem::Mult(*mat, *trial_P);
      delete mat;
      mat = m;
   }

   EliminateTrialVDofs(trial_tdof_list);
   EliminateTestVDofs(test_tdof_list);

   A.Reset(mat, false);
}

void MixedBilinearForm::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   Vector &x, Vector &b,
   OperatorHandle &A,
   Vector &X, Vector &B)
{
   if (ext)
   {
      ext->FormRectangularLinearSystem(trial_tdof_list, test_tdof_list,
                                       x, b, A, X, B);
      return;
   }

   const Operator *Pi = this->GetProlongation();
   const Operator *Po = this->GetOutputProlongation();
   const Operator *Ri = this->GetRestriction();
   InitTVectors(Po, Ri, Pi, x, b, X, B);

   if (!mat_e)
   {
      FormRectangularSystemMatrix(trial_tdof_list, test_tdof_list,
                                  A); // Set A = mat_e
   }
   // Eliminate essential BCs with B -= Ab xb
   EliminateTrialVDofsInRHS(trial_tdof_list, X, B);

   B.SetSubVector(test_tdof_list, 0.0);
}

void MixedBilinearForm::Update()
{
   delete mat;
   mat = NULL;
   delete mat_e;
   mat_e = NULL;
   height = test_fes->GetVSize();
   width = trial_fes->GetVSize();
   if (ext) { ext->Update(); }
}

MixedBilinearForm::~MixedBilinearForm()
{
   if (mat) { delete mat; }
   if (mat_e) { delete mat_e; }
   if (!extern_bfs)
   {
      int i;
      for (i = 0; i < domain_integs.Size(); i++) { delete domain_integs[i]; }
      for (i = 0; i < boundary_integs.Size(); i++)
      { delete boundary_integs[i]; }
      for (i = 0; i < interior_face_integs.Size(); i++)
      { delete interior_face_integs[i]; }
      for (i = 0; i < boundary_face_integs.Size(); i++)
      { delete boundary_face_integs[i]; }
      for (i = 0; i < trace_face_integs.Size(); i++)
      { delete trace_face_integs[i]; }
      for (i = 0; i < boundary_trace_face_integs.Size(); i++)
      { delete boundary_trace_face_integs[i]; }
   }
   delete ext;
}

void DiscreteLinearOperator::SetAssemblyLevel(AssemblyLevel assembly_level)
{
   if (ext)
   {
      MFEM_ABORT("the assembly level has already been set!");
   }
   assembly = assembly_level;
   switch (assembly)
   {
      case AssemblyLevel::LEGACY:
      case AssemblyLevel::FULL:
         // Use the original implementation for now
         break;
      case AssemblyLevel::ELEMENT:
         mfem_error("Element assembly not supported yet... stay tuned!");
         break;
      case AssemblyLevel::PARTIAL:
         ext = new PADiscreteLinearOperatorExtension(this);
         break;
      case AssemblyLevel::NONE:
         mfem_error("Matrix-free action not supported yet... stay tuned!");
         break;
      default:
         mfem_error("Unknown assembly level");
   }
}

void DiscreteLinearOperator::Assemble(int skip_zeros)
{
   if (ext)
   {
      ext->Assemble();
      return;
   }

   ElementTransformation *eltrans;
   DofTransformation * dom_dof_trans;
   DofTransformation * ran_dof_trans;
   DenseMatrix elmat;

   Mesh *mesh = test_fes->GetMesh();

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (domain_integs.Size())
   {
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(domain_integs_marker[k]->Size() ==
                        (mesh->attributes.Size() ? mesh->attributes.Max() : 0),
                        "invalid element marker for domain integrator #"
                        << k << ", counting from zero");
         }
      }

      for (int i = 0; i < test_fes->GetNE(); i++)
      {
         const int elem_attr = mesh->GetAttribute(i);
         dom_dof_trans = trial_fes->GetElementVDofs(i, trial_vdofs);
         ran_dof_trans = test_fes->GetElementVDofs(i, test_vdofs);
         eltrans = test_fes->GetElementTransformation(i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int k = 0; k < domain_integs.Size(); k++)
         {
            if (domain_integs_marker[k] == NULL ||
                (*(domain_integs_marker[k]))[elem_attr-1] == 1)
            {
               domain_integs[k]->AssembleElementMatrix2(*trial_fes->GetFE(i),
                                                        *test_fes->GetFE(i),
                                                        *eltrans, elemmat);
               elmat += elemmat;
            }
         }
         if (ran_dof_trans || dom_dof_trans)
         {
            TransformPrimal(ran_dof_trans, dom_dof_trans, elemmat);
         }
         mat->SetSubMatrix(test_vdofs, trial_vdofs, elemmat, skip_zeros);
      }
   }

   if (trace_face_integs.Size())
   {
      const int nfaces = test_fes->GetMesh()->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         trial_fes->GetFaceVDofs(i, trial_vdofs);
         test_fes->GetFaceVDofs(i, test_vdofs);
         eltrans = test_fes->GetMesh()->GetFaceTransformation(i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int k = 0; k < trace_face_integs.Size(); k++)
         {
            trace_face_integs[k]->AssembleElementMatrix2(*trial_fes->GetFaceElement(i),
                                                         *test_fes->GetFaceElement(i),
                                                         *eltrans, elemmat);
            elmat += elemmat;
         }
         mat->SetSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
      }
   }
}

}
