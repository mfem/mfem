// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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

   if (fbfi.Size() > 0)
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
   double *data = Memory<double>(I[height]);

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

   assembly = AssemblyLevel::FULL;
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

   assembly = AssemblyLevel::FULL;
   batch = 1;
   ext = NULL;

   // Copy the pointers to the integrators
   dbfi = bf->dbfi;

   bbfi = bf->bbfi;
   bbfi_marker = bf->bbfi_marker;

   fbfi = bf->fbfi;

   bfbfi = bf->bfbfi;
   bfbfi_marker = bf->bfbfi_marker;

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
      case AssemblyLevel::FULL:
         // ext = new FABilinearFormExtension(this);
         // Use the original BilinearForm implementation for now
         break;
      case AssemblyLevel::ELEMENT:
         mfem_error("Element assembly not supported yet... stay tuned!");
         // ext = new EABilinearFormExtension(this);
         break;
      case AssemblyLevel::PARTIAL:
         ext = new PABilinearFormExtension(this);
         break;
      case AssemblyLevel::NONE:
         mfem_error("Matrix-free action not supported yet... stay tuned!");
         // ext = new MFBilinearFormExtension(this);
         break;
      default:
         mfem_error("Unknown assembly level");
   }
}

void BilinearForm::EnableStaticCondensation()
{
   delete static_cond;
   if (assembly != AssemblyLevel::FULL)
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
   if (assembly != AssemblyLevel::FULL)
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

double& BilinearForm::Elem (int i, int j)
{
   return mat -> Elem(i,j);
}

const double& BilinearForm::Elem (int i, int j) const
{
   return mat -> Elem(i,j);
}

MatrixInverse * BilinearForm::Inverse() const
{
   return mat -> Inverse();
}

void BilinearForm::Finalize (int skip_zeros)
{
   if (assembly == AssemblyLevel::FULL)
   {
      if (!static_cond) { mat->Finalize(skip_zeros); }
      if (mat_e) { mat_e->Finalize(skip_zeros); }
      if (static_cond) { static_cond->Finalize(); }
      if (hybridization) { hybridization->Finalize(); }
   }
}

void BilinearForm::AddDomainIntegrator(BilinearFormIntegrator *bfi)
{
   dbfi.Append(bfi);
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bbfi.Append (bfi);
   bbfi_marker.Append(NULL); // NULL marker means apply everywhere
}

void BilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi,
                                          Array<int> &bdr_marker)
{
   bbfi.Append (bfi);
   bbfi_marker.Append(&bdr_marker);
}

void BilinearForm::AddInteriorFaceIntegrator (BilinearFormIntegrator * bfi)
{
   fbfi.Append (bfi);
}

void BilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi)
{
   bfbfi.Append(bfi);
   bfbfi_marker.Append(NULL); // NULL marker means apply everywhere
}

void BilinearForm::AddBdrFaceIntegrator(BilinearFormIntegrator *bfi,
                                        Array<int> &bdr_marker)
{
   bfbfi.Append(bfi);
   bfbfi_marker.Append(&bdr_marker);
}

void BilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   if (element_matrices)
   {
      elmat.SetSize(element_matrices->SizeI(), element_matrices->SizeJ());
      elmat = element_matrices->GetData(i);
      return;
   }

   if (dbfi.Size())
   {
      const FiniteElement &fe = *fes->GetFE(i);
      ElementTransformation *eltrans = fes->GetElementTransformation(i);
      dbfi[0]->AssembleElementMatrix(fe, *eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      fes->GetElementVDofs(i, vdofs);
      elmat.SetSize(vdofs.Size());
      elmat = 0.0;
   }
}

void BilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat)
{
   if (bbfi.Size())
   {
      const FiniteElement &be = *fes->GetBE(i);
      ElementTransformation *eltrans = fes->GetBdrElementTransformation(i);
      bbfi[0]->AssembleElementMatrix(be, *eltrans, elmat);
      for (int k = 1; k < bbfi.Size(); k++)
      {
         bbfi[k]->AssembleElementMatrix(be, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      fes->GetBdrElementVDofs(i, vdofs);
      elmat.SetSize(vdofs.Size());
      elmat = 0.0;
   }
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleElementMatrix(i, elmat, vdofs, skip_zeros);
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &vdofs, int skip_zeros)
{
   fes->GetElementVDofs(i, vdofs);
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
      mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
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
   int i, const DenseMatrix &elmat, Array<int> &vdofs, int skip_zeros)
{
   fes->GetBdrElementVDofs(i, vdofs);
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
      mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
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

   if (dbfi.Size())
   {
      for (int i = 0; i < fes -> GetNE(); i++)
      {
         fes->GetElementVDofs(i, vdofs);
         if (element_matrices)
         {
            elmat_p = &(*element_matrices)(i);
         }
         else
         {
            const FiniteElement &fe = *fes->GetFE(i);
            eltrans = fes->GetElementTransformation(i);
            dbfi[0]->AssembleElementMatrix(fe, *eltrans, elmat);
            for (int k = 1; k < dbfi.Size(); k++)
            {
               dbfi[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
               elmat += elemmat;
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
   }

   if (bbfi.Size())
   {
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bbfi.Size(); k++)
      {
         if (bbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bbfi_marker[k];
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
         fes -> GetBdrElementVDofs (i, vdofs);
         eltrans = fes -> GetBdrElementTransformation (i);
         bbfi[0]->AssembleElementMatrix(be, *eltrans, elmat);
         for (int k = 1; k < bbfi.Size(); k++)
         {
            if (bbfi_marker[k] &&
                (*bbfi_marker[k])[bdr_attr-1] == 0) { continue; }

            bbfi[k]->AssembleElementMatrix(be, *eltrans, elemmat);
            elmat += elemmat;
         }
         if (!static_cond)
         {
            mat->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
            if (hybridization)
            {
               hybridization->AssembleBdrMatrix(i, elmat);
            }
         }
         else
         {
            static_cond->AssembleBdrMatrix(i, elmat);
         }
      }
   }

   if (fbfi.Size())
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
            for (int k = 0; k < fbfi.Size(); k++)
            {
               fbfi[k] -> AssembleFaceMatrix (*fes -> GetFE (tr -> Elem1No),
                                              *fes -> GetFE (tr -> Elem2No),
                                              *tr, elemmat);
               mat -> AddSubMatrix (vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (bfbfi.Size())
   {
      FaceElementTransformations *tr;
      const FiniteElement *fe1, *fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bfbfi.Size(); k++)
      {
         if (bfbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bfbfi_marker[k];
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
            for (int k = 0; k < bfbfi.Size(); k++)
            {
               if (bfbfi_marker[k] &&
                   (*bfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

               bfbfi[k] -> AssembleFaceMatrix (*fe1, *fe2, *tr, elemmat);
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
   if (ext)
   {
      MFEM_ASSERT(diag.Size() == fes->GetTrueVSize(),
                  "Vector for holding diagonal has wrong size!");
      const Operator *P = fes->GetProlongationMatrix();
      if (!IsIdentityProlongation(P))
      {
         Vector local_diag(P->Height());
         ext->AssembleDiagonal(local_diag);
         P->MultTranspose(local_diag, diag);
      }
      else
      {
         ext->AssembleDiagonal(diag);
      }
   }
   else
   {
      MFEM_ABORT("Not implemented. Maybe assemble your bilinear form into a "
                 "matrix and use SparseMatrix::GetDiag?");
   }
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
         X.NewMemoryAndSize(x.GetMemory(), x.Size(), false);
         B.NewMemoryAndSize(b.GetMemory(), b.Size(), false);
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
   if (element_matrices || dbfi.Size() == 0 || fes->GetNE() == 0)
   {
      return;
   }

   int num_elements = fes->GetNE();
   int num_dofs_per_el = fes->GetFE(0)->GetDof() * fes->GetVDim();

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

      dbfi[0]->AssembleElementMatrix(fe, eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         // note: some integrators may not be thread-safe
         dbfi[k]->AssembleElementMatrix(fe, eltrans, tmp);
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
                                             double value)
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

void BilinearForm::EliminateVDofs(const Array<int> &vdofs,
                                  const Vector &sol, Vector &rhs,
                                  DiagonalPolicy dpolicy)
{
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
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

void BilinearForm::EliminateVDofs(const Array<int> &vdofs,
                                  DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(height);
   }

   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
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
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowCol (i, dpolicy);
      }
}

void BilinearForm::EliminateEssentialBCFromDofsDiag (const Array<int> &ess_dofs,
                                                     double value)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat -> EliminateRowColDiag (i, value);
      }
}

void BilinearForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
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
      for (k=0; k < dbfi.Size(); k++) { delete dbfi[k]; }
      for (k=0; k < bbfi.Size(); k++) { delete bbfi[k]; }
      for (k=0; k < fbfi.Size(); k++) { delete fbfi[k]; }
      for (k=0; k < bfbfi.Size(); k++) { delete bfbfi[k]; }
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
   assembly = AssemblyLevel::FULL;
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
   dbfi = mbf->dbfi;
   bbfi = mbf->bbfi;
   tfbfi = mbf->tfbfi;
   btfbfi = mbf->btfbfi;

   bbfi_marker = mbf->bbfi_marker;
   btfbfi_marker = mbf->btfbfi_marker;

   assembly = AssemblyLevel::FULL;
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

double & MixedBilinearForm::Elem (int i, int j)
{
   return (*mat)(i, j);
}

const double & MixedBilinearForm::Elem (int i, int j) const
{
   return (*mat)(i, j);
}

void MixedBilinearForm::Mult(const Vector & x, Vector & y) const
{
   y = 0.0;
   AddMult(x, y);
}

void MixedBilinearForm::AddMult(const Vector & x, Vector & y,
                                const double a) const
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
                                         const double a) const
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
   if (assembly != AssemblyLevel::FULL)
   {
      MFEM_WARNING("MixedBilinearForm::Inverse not possible with this assembly level!");
      return NULL;
   }
   else
   {
      return mat -> Inverse ();
   }
}

void MixedBilinearForm::Finalize (int skip_zeros)
{
   if (assembly == AssemblyLevel::FULL)
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

void MixedBilinearForm::AddDomainIntegrator (BilinearFormIntegrator * bfi)
{
   dbfi.Append (bfi);
}

void MixedBilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi)
{
   bbfi.Append (bfi);
   bbfi_marker.Append(NULL); // NULL marker means apply everywhere
}

void MixedBilinearForm::AddBoundaryIntegrator (BilinearFormIntegrator * bfi,
                                               Array<int> &bdr_marker)
{
   bbfi.Append (bfi);
   bbfi_marker.Append(&bdr_marker);
}

void MixedBilinearForm::AddTraceFaceIntegrator (BilinearFormIntegrator * bfi)
{
   tfbfi.Append (bfi);
}

void MixedBilinearForm::AddBdrTraceFaceIntegrator(BilinearFormIntegrator *bfi)
{
   btfbfi.Append(bfi);
   btfbfi_marker.Append(NULL); // NULL marker means apply everywhere
}

void MixedBilinearForm::AddBdrTraceFaceIntegrator(BilinearFormIntegrator *bfi,
                                                  Array<int> &bdr_marker)
{
   btfbfi.Append(bfi);
   btfbfi_marker.Append(&bdr_marker);
}

void MixedBilinearForm::Assemble (int skip_zeros)
{
   if (ext)
   {
      ext->Assemble();
      return;
   }

   Array<int> tr_vdofs, te_vdofs;
   ElementTransformation *eltrans;
   DenseMatrix elemmat;

   Mesh *mesh = test_fes -> GetMesh();

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (dbfi.Size())
   {
      for (int i = 0; i < test_fes -> GetNE(); i++)
      {
         trial_fes -> GetElementVDofs (i, tr_vdofs);
         test_fes  -> GetElementVDofs (i, te_vdofs);
         eltrans = test_fes -> GetElementTransformation (i);
         for (int k = 0; k < dbfi.Size(); k++)
         {
            dbfi[k] -> AssembleElementMatrix2 (*trial_fes -> GetFE(i),
                                               *test_fes  -> GetFE(i),
                                               *eltrans, elemmat);
            mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (bbfi.Size())
   {
      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < bbfi.Size(); k++)
      {
         if (bbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *bbfi_marker[k];
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

         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         test_fes  -> GetBdrElementVDofs (i, te_vdofs);
         eltrans = test_fes -> GetBdrElementTransformation (i);
         for (int k = 0; k < bbfi.Size(); k++)
         {
            if (bbfi_marker[k] &&
                (*bbfi_marker[k])[bdr_attr-1] == 0) { continue; }

            bbfi[k] -> AssembleElementMatrix2 (*trial_fes -> GetBE(i),
                                               *test_fes  -> GetBE(i),
                                               *eltrans, elemmat);
            mat -> AddSubMatrix (te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (tfbfi.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> te_vdofs2;
      const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

      int nfaces = mesh->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         ftr = mesh->GetFaceElementTransformations(i);
         trial_fes->GetFaceVDofs(i, tr_vdofs);
         test_fes->GetElementVDofs(ftr->Elem1No, te_vdofs);
         trial_face_fe = trial_fes->GetFaceElement(i);
         test_fe1 = test_fes->GetFE(ftr->Elem1No);
         if (ftr->Elem2No >= 0)
         {
            test_fes->GetElementVDofs(ftr->Elem2No, te_vdofs2);
            te_vdofs.Append(te_vdofs2);
            test_fe2 = test_fes->GetFE(ftr->Elem2No);
         }
         else
         {
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            test_fe2 = test_fe1;
         }
         for (int k = 0; k < tfbfi.Size(); k++)
         {
            tfbfi[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
                                         *ftr, elemmat);
            mat->AddSubMatrix(te_vdofs, tr_vdofs, elemmat, skip_zeros);
         }
      }
   }

   if (btfbfi.Size())
   {
      FaceElementTransformations *ftr;
      Array<int> te_vdofs2;
      const FiniteElement *trial_face_fe, *test_fe1, *test_fe2;

      // Which boundary attributes need to be processed?
      Array<int> bdr_attr_marker(mesh->bdr_attributes.Size() ?
                                 mesh->bdr_attributes.Max() : 0);
      bdr_attr_marker = 0;
      for (int k = 0; k < btfbfi.Size(); k++)
      {
         if (btfbfi_marker[k] == NULL)
         {
            bdr_attr_marker = 1;
            break;
         }
         Array<int> &bdr_marker = *btfbfi_marker[k];
         MFEM_ASSERT(bdr_marker.Size() == bdr_attr_marker.Size(),
                     "invalid boundary marker for boundary trace face integrator #"
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

         ftr = mesh->GetBdrFaceTransformations(i);
         if (ftr)
         {
            trial_fes->GetFaceVDofs(i, tr_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, te_vdofs);
            trial_face_fe = trial_fes->GetFaceElement(i);
            test_fe1 = test_fes->GetFE(ftr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            test_fe2 = test_fe1;
            for (int k = 0; k < btfbfi.Size(); k++)
            {
               if (btfbfi_marker[k] &&
                   (*btfbfi_marker[k])[bdr_attr-1] == 0) { continue; }

               btfbfi[k]->AssembleFaceMatrix(*trial_face_fe, *test_fe1, *test_fe2,
                                             *ftr, elemmat);
               mat->AddSubMatrix(te_vdofs, tr_vdofs, elemmat, skip_zeros);
            }
         }
      }
   }
}

void MixedBilinearForm::ConformingAssemble()
{
   if (assembly != AssemblyLevel::FULL)
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


void MixedBilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat)
{
   if (dbfi.Size())
   {
      const FiniteElement &trial_fe = *trial_fes->GetFE(i);
      const FiniteElement &test_fe = *test_fes->GetFE(i);
      ElementTransformation *eltrans = test_fes->GetElementTransformation(i);
      dbfi[0]->AssembleElementMatrix2(trial_fe, test_fe, *eltrans, elmat);
      for (int k = 1; k < dbfi.Size(); k++)
      {
         dbfi[k]->AssembleElementMatrix2(trial_fe, test_fe, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      trial_fes->GetElementVDofs(i, trial_vdofs);
      test_fes->GetElementVDofs(i, test_vdofs);
      elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
      elmat = 0.0;
   }
}

void MixedBilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat)
{
   if (bbfi.Size())
   {
      const FiniteElement &trial_be = *trial_fes->GetBE(i);
      const FiniteElement &test_be = *test_fes->GetBE(i);
      ElementTransformation *eltrans = test_fes->GetBdrElementTransformation(i);
      bbfi[0]->AssembleElementMatrix2(trial_be, test_be, *eltrans, elmat);
      for (int k = 1; k < bbfi.Size(); k++)
      {
         bbfi[k]->AssembleElementMatrix2(trial_be, test_be, *eltrans, elemmat);
         elmat += elemmat;
      }
   }
   else
   {
      trial_fes->GetBdrElementVDofs(i, trial_vdofs);
      test_fes->GetBdrElementVDofs(i, test_vdofs);
      elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
      elmat = 0.0;
   }
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &trial_vdofs,
   Array<int> &test_vdofs, int skip_zeros)
{
   trial_fes->GetElementVDofs(i, trial_vdofs);
   test_fes->GetElementVDofs(i, test_vdofs);
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   AssembleBdrElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, Array<int> &trial_vdofs,
   Array<int> &test_vdofs, int skip_zeros)
{
   trial_fes->GetBdrElementVDofs(i, trial_vdofs);
   test_fes->GetBdrElementVDofs(i, test_vdofs);
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
}

void MixedBilinearForm::EliminateTrialDofs (
   const Array<int> &bdr_attr_is_ess, const Vector &sol, Vector &rhs )
{
   int i, j, k;
   Array<int> tr_vdofs, cols_marker (trial_fes -> GetVSize());

   cols_marker = 0;
   for (i = 0; i < trial_fes -> GetNBE(); i++)
      if (bdr_attr_is_ess[trial_fes -> GetBdrAttribute (i)-1])
      {
         trial_fes -> GetBdrElementVDofs (i, tr_vdofs);
         for (j = 0; j < tr_vdofs.Size(); j++)
         {
            if ( (k = tr_vdofs[j]) < 0 )
            {
               k = -1-k;
            }
            cols_marker[k] = 1;
         }
      }
   mat -> EliminateCols (cols_marker, &sol, &rhs);
}

void MixedBilinearForm::EliminateEssentialBCFromTrialDofs (
   const Array<int> &marked_vdofs, const Vector &sol, Vector &rhs)
{
   mat -> EliminateCols (marked_vdofs, &sol, &rhs);
}

void MixedBilinearForm::EliminateTestDofs (const Array<int> &bdr_attr_is_ess)
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

void MixedBilinearForm::FormRectangularSystemMatrix(const Array<int>
                                                    &trial_tdof_list,
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

   if (test_P) // TODO: Must actually check for trial_P too
   {
      SparseMatrix *m = RAP(*test_P, *mat, *trial_P);
      delete mat;
      mat = m;
   }

   Array<int> ess_trial_tdof_marker, ess_test_tdof_marker;
   FiniteElementSpace::ListToMarker(trial_tdof_list, trial_fes->GetTrueVSize(),
                                    ess_trial_tdof_marker);
   FiniteElementSpace::ListToMarker(test_tdof_list, test_fes->GetTrueVSize(),
                                    ess_test_tdof_marker);

   mat_e = new SparseMatrix(mat->Height(), mat->Width());
   mat->EliminateCols(ess_trial_tdof_marker, *mat_e);

   for (int i=0; i<test_tdof_list.Size(); ++i)
   {
      mat->EliminateRow(test_tdof_list[i]);
   }
   mat_e->Finalize();
   A.Reset(mat, false);
}

void MixedBilinearForm::FormRectangularLinearSystem(const Array<int>
                                                    &trial_tdof_list,
                                                    const Array<int> &test_tdof_list,
                                                    Vector &x, Vector &b,
                                                    OperatorHandle &A,
                                                    Vector &X, Vector &B)
{
   if (ext)
   {
      ext->FormRectangularLinearSystem(trial_tdof_list, test_tdof_list, x, b, A, X,
                                       B);
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
   mat_e->AddMult(X, B, -1.0);

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
      for (i = 0; i < dbfi.Size(); i++) { delete dbfi[i]; }
      for (i = 0; i < bbfi.Size(); i++) { delete bbfi[i]; }
      for (i = 0; i < tfbfi.Size(); i++) { delete tfbfi[i]; }
      for (i = 0; i < btfbfi.Size(); i++) { delete btfbfi[i]; }
   }
   delete ext;
}


void DiscreteLinearOperator::Assemble(int skip_zeros)
{
   Array<int> dom_vdofs, ran_vdofs;
   ElementTransformation *T;
   const FiniteElement *dom_fe, *ran_fe;
   DenseMatrix totelmat, elmat;

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (dbfi.Size() > 0)
   {
      for (int i = 0; i < test_fes->GetNE(); i++)
      {
         trial_fes->GetElementVDofs(i, dom_vdofs);
         test_fes->GetElementVDofs(i, ran_vdofs);
         T = test_fes->GetElementTransformation(i);
         dom_fe = trial_fes->GetFE(i);
         ran_fe = test_fes->GetFE(i);

         dbfi[0]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, totelmat);
         for (int j = 1; j < dbfi.Size(); j++)
         {
            dbfi[j]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, elmat);
            totelmat += elmat;
         }
         mat->SetSubMatrix(ran_vdofs, dom_vdofs, totelmat, skip_zeros);
      }
   }

   if (tfbfi.Size())
   {
      const int nfaces = test_fes->GetMesh()->GetNumFaces();
      for (int i = 0; i < nfaces; i++)
      {
         trial_fes->GetFaceVDofs(i, dom_vdofs);
         test_fes->GetFaceVDofs(i, ran_vdofs);
         T = test_fes->GetMesh()->GetFaceTransformation(i);
         dom_fe = trial_fes->GetFaceElement(i);
         ran_fe = test_fes->GetFaceElement(i);

         tfbfi[0]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, totelmat);
         for (int j = 1; j < tfbfi.Size(); j++)
         {
            tfbfi[j]->AssembleElementMatrix2(*dom_fe, *ran_fe, *T, elmat);
            totelmat += elmat;
         }
         mat->SetSubMatrix(ran_vdofs, dom_vdofs, totelmat, skip_zeros);
      }
   }
}

}
