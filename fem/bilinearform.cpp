// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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

BilinearForm::BilinearForm(FiniteElementSpace *f)
   : Matrix(f->GetVSize())
{
   fes = f;
   sequence = f->GetSequence();
   mat = mat_e = NULL;
   extern_bfs = 0;
   static_cond = NULL;
   hybridization = NULL;
   diag_policy = DIAG_KEEP;
   assembly = AssemblyLevel::LEGACY;
   ext = NULL;
}

BilinearForm::BilinearForm(FiniteElementSpace *f, BilinearForm *bf)
   : Matrix(f->GetVSize())
{
   fes = f;
   sequence = f->GetSequence();
   mat = mat_e = NULL;
   extern_bfs = 1;
   static_cond = NULL;
   hybridization = NULL;
   diag_policy = DIAG_KEEP;
   assembly = AssemblyLevel::LEGACY;
   ext = NULL;

   // Copy the pointers to the integrators
   domain_integs = bf->domain_integs;

   boundary_integs = bf->boundary_integs;
   boundary_integs_marker = bf->boundary_integs_marker;

   interior_face_integs = bf->interior_face_integs;

   boundary_face_integs = bf->boundary_face_integs;
   boundary_face_integs_marker = bf->boundary_face_integs_marker;
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
         SetDiagonalPolicy(DIAG_ONE); // Only diagonal policy supported on device
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

double &BilinearForm::Elem(int i, int j)
{
   return mat->Elem(i,j);
}

const double &BilinearForm::Elem(int i, int j) const
{
   return mat->Elem(i,j);
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

void BilinearForm::AddMult(const Vector &x, Vector &y, const double a) const
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

void BilinearForm::MultTranspose(const Vector &x, Vector &y) const
{
   if (ext)
   {
      ext->MultTranspose(x, y);
   }
   else
   {
      mat->MultTranspose(x, y);
   }
}

void BilinearForm::AddMultTranspose(const Vector &x, Vector &y,
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

MatrixInverse *BilinearForm::Inverse() const
{
   return mat->Inverse();
}

void BilinearForm::Finalize(int skip_zeros)
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

void BilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi)
{
   boundary_integs.Append(bfi);
   boundary_integs_marker.Append(NULL); // NULL marker means apply everywhere
}

void BilinearForm::AddBoundaryIntegrator(BilinearFormIntegrator *bfi,
                                         Array<int> &bdr_marker)
{
   boundary_integs.Append(bfi);
   boundary_integs_marker.Append(&bdr_marker);
}

void BilinearForm::AddInteriorFaceIntegrator(BilinearFormIntegrator *bfi)
{
   interior_face_integs.Append(bfi);
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

void BilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat,
                                        Array<int> &vdofs)
{
   DofTransformation *doftrans = fes->GetElementVDofs(i, vdofs);
   elmat.SetSize(vdofs.Size());
   elmat = 0.0;
   if (domain_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();
      ElementTransformation *eltrans = mesh->GetElementTransformation(i);
      int elem_attr = mesh->GetAttribute(i);
#ifdef MFEM_DEBUG
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
#endif
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         if (domain_integs_marker[k] == NULL ||
             (*(domain_integs_marker[k]))[elem_attr-1] == 1)
         {
            domain_integs[k]->AssembleElementMatrix(*fes->GetFE(i),
                                                    *eltrans, elemmat);
            elmat += elemmat;
         }
      }
      if (doftrans)
      {
         doftrans->TransformDual(elmat);
      }
   }
}

void BilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat,
                                           Array<int> &vdofs)
{
   DofTransformation *doftrans = fes->GetBdrElementVDofs(i, vdofs);
   elmat.SetSize(vdofs.Size());
   elmat = 0.0;
   if (boundary_integs.Size())
   {
      Mesh *mesh = fes->GetMesh();
      ElementTransformation *eltrans = mesh->GetBdrElementTransformation(i);
      int bdr_attr = mesh->GetBdrAttribute(i);
#ifdef MFEM_DEBUG
      for (int k = 0; k < boundary_integs.Size(); k++)
      {
         if (boundary_integs_marker[k] != NULL)
         {
            MFEM_VERIFY(boundary_integs_marker[k]->Size() ==
                        (mesh->bdr_attributes.Size() ? mesh->bdr_attributes.Max() : 0),
                        "invalid element marker for bdr integrator #"
                        << k << ", counting from zero");
         }
      }
#endif
      for (int k = 0; k < boundary_integs.Size(); k++)
      {
         if (boundary_integs_marker[k] == NULL ||
             (*(boundary_integs_marker[k]))[bdr_attr-1] == 1)
         {
            boundary_integs[k]->AssembleElementMatrix(*fes->GetBE(i),
                                                      *eltrans, elemmat);
            elmat += elemmat;
         }
      }
      if (doftrans)
      {
         doftrans->TransformDual(elmat);
      }
   }
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   Array<int> vdofs;
   fes->GetElementVDofs(i, vdofs);
   AssembleElementMatrix(i, elmat, vdofs, skip_zeros);
}

void BilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, const Array<int> &vdofs, int skip_zeros)
{
   if (static_cond)
   {
      static_cond->AssembleMatrix(i, elmat);
   }
   else
   {
      if (mat == NULL)
      {
         mat = new SparseMatrix(height);
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
   Array<int> vdofs;
   fes->GetBdrElementVDofs(i, vdofs);
   AssembleBdrElementMatrix(i, elmat, vdofs, skip_zeros);
}

void BilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, const Array<int> &vdofs, int skip_zeros)
{
   if (static_cond)
   {
      static_cond->AssembleBdrMatrix(i, elmat);
   }
   else
   {
      if (mat == NULL)
      {
         mat = new SparseMatrix(height);
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

   Mesh *mesh = fes->GetMesh();
   DenseMatrix elmat;
   Array<int> vdofs;

   if (mat == NULL)
   {
      mat = new SparseMatrix(height);
   }

   if (domain_integs.Size())
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         ComputeElementMatrix(i, elmat, vdofs);
         AssembleElementMatrix(i, elmat, vdofs, skip_zeros);
      }
   }

   if (boundary_integs.Size())
   {
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         ComputeBdrElementMatrix(i, elmat, vdofs);
         AssembleBdrElementMatrix(i, elmat, vdofs, skip_zeros);
      }
   }

   if (interior_face_integs.Size())
   {
      Array<int> vdofs2;
      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(i);
         if (tr != NULL)
         {
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            fes->GetElementVDofs(tr->Elem2No, vdofs2);
            vdofs.Append(vdofs2);
            for (int k = 0; k < interior_face_integs.Size(); k++)
            {
               interior_face_integs[k]->AssembleFaceMatrix(*fes->GetFE(tr->Elem1No),
                                                           *fes->GetFE(tr->Elem2No),
                                                           *tr, elemmat);
               mat->AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
            }
         }
      }
   }

   if (boundary_face_integs.Size())
   {
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         FaceElementTransformations *tr = mesh->GetBdrFaceTransformations(i);
         if (tr != NULL)
         {
            int bdr_attr = mesh->GetBdrAttribute(i);
            const FiniteElement *fe1 = fes->GetFE(tr->Elem1No);
            // The fe2 object is really a dummy and not used on the boundaries,
            // but we can't dereference a NULL pointer, and we don't want to
            // actually make a fake element.
            const FiniteElement *fe2 = fe1;
            fes->GetElementVDofs(tr->Elem1No, vdofs);
            for (int k = 0; k < boundary_face_integs.Size(); k++)
            {
               if (boundary_face_integs_marker[k] == NULL ||
                   (*boundary_face_integs_marker[k])[bdr_attr-1] == 1)
               {
                  boundary_face_integs[k]->AssembleFaceMatrix(*fe1, *fe2, *tr,
                                                              elemmat);
                  mat->AddSubMatrix(vdofs, vdofs, elemmat, skip_zeros);
               }
            }
         }
      }
   }
}

void BilinearForm::ConformingAssemble()
{
   // Do not remove zero entries to preserve the symmetric structure of the
   // matrix which in turn will give rise to symmetric structure in the new
   // matrix. This ensures that subsequent calls to EliminateRowCol will work
   // correctly.
   MFEM_ASSERT(mat, "the BilinearForm is not assembled");
   const int remove_zeros = 0;
   Finalize(remove_zeros);

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
      Operator *oper;
      ext->FormLinearSystem(ess_tdof_list, x, b, oper, X, B, copy_interior);
      if (assembly == AssemblyLevel::FULL)
      {
         delete oper;
         FormSystemMatrix(ess_tdof_list, A);
      }
      else
      {
         A.Reset(oper);
      }
      return;
   }

   // Finish the matrix assembly and perform BC elimination, storing the
   // eliminated part of the matrix.
   FormSystemMatrix(ess_tdof_list, A);

   const SparseMatrix *P = fes->GetConformingProlongation();

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
      if (assembly == AssemblyLevel::FULL)
      {
         // Always does `DIAG_ONE` policy to be consistent with
         // `Operator::FormConstrainedSystemOperator`.
         MFEM_VERIFY(diag_policy == DiagonalPolicy::DIAG_ONE,
                     "Only DiagonalPolicy::DIAG_ONE supported with"
                     " FABilinearFormExtension.");
         ConformingAssemble();
         mat->EliminateBC(ess_tdof_list, DiagonalPolicy::DIAG_ONE);
         A.Reset(mat, false);
      }
      else
      {
         Operator *oper;
         ext->FormSystemOperator(ess_tdof_list, oper);
         A.Reset(oper);
      }
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
         ConformingAssemble();
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

void BilinearForm::EliminateEssentialBCDiag(const Array<int> &bdr_attr_is_ess,
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
   vdofs.HostRead();
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if (vdof >= 0)
      {
         mat->EliminateRowCol(vdof, sol(vdof), rhs, dpolicy);
      }
      else
      {
         mat->EliminateRowCol(-1-vdof, sol(-1-vdof), rhs, dpolicy);
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

   vdofs.HostRead();
   for (int i = 0; i < vdofs.Size(); i++)
   {
      int vdof = vdofs[i];
      if (vdof >= 0)
      {
         mat->EliminateRowCol(vdof, *mat_e, dpolicy);
      }
      else
      {
         mat->EliminateRowCol(-1-vdof, *mat_e, dpolicy);
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
   {
      if (ess_dofs[i] < 0)
      {
         mat->EliminateRowCol(i, sol(i), rhs, dpolicy);
      }
   }
}

void BilinearForm::EliminateEssentialBCFromDofs(const Array<int> &ess_dofs,
                                                DiagonalPolicy dpolicy)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
   {
      if (ess_dofs[i] < 0)
      {
         mat->EliminateRowCol(i, dpolicy);
      }
   }
}

void BilinearForm::EliminateEssentialBCFromDofsDiag(const Array<int> &ess_dofs,
                                                    double value)
{
   MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

   for (int i = 0; i < ess_dofs.Size(); i++)
      if (ess_dofs[i] < 0)
      {
         mat->EliminateRowColDiag(i, value);
      }
}

void BilinearForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
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

BilinearForm::~BilinearForm()
{
   delete mat_e;
   delete mat;
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

MixedBilinearForm::MixedBilinearForm(FiniteElementSpace *tr_fes,
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

MixedBilinearForm::MixedBilinearForm(FiniteElementSpace *tr_fes,
                                     FiniteElementSpace *te_fes,
                                     MixedBilinearForm *mbf)
   : Matrix(te_fes->GetVSize(), tr_fes->GetVSize())
{
   trial_fes = tr_fes;
   test_fes = te_fes;
   mat = NULL;
   mat_e = NULL;
   extern_bfs = 1;
   ext = NULL;
   assembly = AssemblyLevel::LEGACY;
   ext = NULL;

   // Copy the pointers to the integrators
   domain_integs = mbf->domain_integs;
   boundary_integs = mbf->boundary_integs;
   trace_face_integs = mbf->trace_face_integs;
   boundary_trace_face_integs = mbf->boundary_trace_face_integs;

   boundary_integs_marker = mbf->boundary_integs_marker;
   boundary_trace_face_integs_marker = mbf->boundary_trace_face_integs_marker;
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
         ext = new MFMixedBilinearFormExtension(this);
         break;
      default:
         mfem_error("Unknown assembly level");
   }
}

double &MixedBilinearForm::Elem(int i, int j)
{
   return (*mat)(i, j);
}

const double &MixedBilinearForm::Elem(int i, int j) const
{
   return (*mat)(i, j);
}

void MixedBilinearForm::Mult(const Vector &x, Vector &y) const
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

void MixedBilinearForm::AddMult(const Vector &x, Vector &y,
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

void MixedBilinearForm::MultTranspose(const Vector &x, Vector &y) const
{
   if (ext)
   {
      ext->MultTranspose(x, y);
   }
   else
   {
      mat->MultTranspose(x, y);
   }
}

void MixedBilinearForm::AddMultTranspose(const Vector &x, Vector &y,
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

MatrixInverse *MixedBilinearForm::Inverse() const
{
   if (assembly != AssemblyLevel::LEGACY)
   {
      MFEM_WARNING("MixedBilinearForm::Inverse not possible with this "
                   "assembly level!");
      return NULL;
   }
   else
   {
      return mat->Inverse();
   }
}

void MixedBilinearForm::Finalize(int skip_zeros)
{
   if (assembly == AssemblyLevel::LEGACY)
   {
      mat->Finalize(skip_zeros);
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

void MixedBilinearForm::AddTraceFaceIntegrator(BilinearFormIntegrator *bfi)
{
   trace_face_integs.Append(bfi);
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

void MixedBilinearForm::ComputeElementMatrix(int i, DenseMatrix &elmat,
                                             Array<int> &trial_vdofs,
                                             Array<int> &test_vdofs)
{
   DofTransformation *dom_dof_trans = trial_fes->GetElementVDofs(i, trial_vdofs);
   DofTransformation *ran_dof_trans = test_fes->GetElementVDofs(i, test_vdofs);
   elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
   elmat = 0.0;
   if (domain_integs.Size())
   {
      Mesh *mesh = test_fes->GetMesh();
      ElementTransformation *eltrans = mesh->GetElementTransformation(i);
      for (int k = 0; k < domain_integs.Size(); k++)
      {
         domain_integs[k]->AssembleElementMatrix2(*trial_fes->GetFE(i),
                                                  *test_fes->GetFE(i),
                                                  *eltrans, elemmat);
         elmat += elemmat;
      }
      if (ran_dof_trans || dom_dof_trans)
      {
         TransformDual(ran_dof_trans, dom_dof_trans, elmat);
      }
   }
}

void MixedBilinearForm::ComputeBdrElementMatrix(int i, DenseMatrix &elmat,
                                                Array<int> &trial_vdofs,
                                                Array<int> &test_vdofs)
{
   DofTransformation *dom_dof_trans = trial_fes->GetBdrElementVDofs(i,
                                                                    trial_vdofs);
   DofTransformation *ran_dof_trans = test_fes->GetBdrElementVDofs(i, test_vdofs);
   elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
   elmat = 0.0;
   if (boundary_integs.Size())
   {
      Mesh *mesh = test_fes->GetMesh();
      ElementTransformation *eltrans = mesh->GetBdrElementTransformation(i);
      int bdr_attr = mesh->GetBdrAttribute(i);
      for (int k = 0; k < boundary_integs.Size(); k++)
      {
         if (boundary_integs_marker[k] == NULL ||
             (*boundary_integs_marker[k])[bdr_attr-1] == 1)
         {
            boundary_integs[k]->AssembleElementMatrix2(*trial_fes->GetBE(i),
                                                       *test_fes->GetBE(i),
                                                       *eltrans, elemmat);
            elmat += elemmat;
         }
      }
      if (ran_dof_trans || dom_dof_trans)
      {
         TransformDual(ran_dof_trans, dom_dof_trans, elmat);
      }
   }
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   Array<int> trial_vdofs, test_vdofs;
   trial_fes->GetElementVDofs(i, trial_vdofs);
   test_fes->GetElementVDofs(i, test_vdofs);
   AssembleElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleElementMatrix(
   int i, const DenseMatrix &elmat, const Array<int> &trial_vdofs,
   const Array<int> &test_vdofs, int skip_zeros)
{
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, int skip_zeros)
{
   Array<int> trial_vdofs, test_vdofs;
   trial_fes->GetBdrElementVDofs(i, trial_vdofs);
   test_fes->GetBdrElementVDofs(i, test_vdofs);
   AssembleBdrElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
}

void MixedBilinearForm::AssembleBdrElementMatrix(
   int i, const DenseMatrix &elmat, const Array<int> &trial_vdofs,
   const Array<int> &test_vdofs, int skip_zeros)
{
   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }
   mat->AddSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
}

void MixedBilinearForm::Assemble(int skip_zeros)
{
   if (ext)
   {
      ext->Assemble();
      return;
   }

   Mesh *mesh = test_fes->GetMesh();
   DenseMatrix elmat;
   Array<int> trial_vdofs, test_vdofs;

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (domain_integs.Size())
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         ComputeElementMatrix(i, elmat, trial_vdofs, test_vdofs);
         AssembleElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
      }
   }

   if (boundary_integs.Size())
   {
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         ComputeBdrElementMatrix(i, elmat, trial_vdofs, test_vdofs);
         AssembleBdrElementMatrix(i, elmat, trial_vdofs, test_vdofs, skip_zeros);
      }
   }

   if (trace_face_integs.Size())
   {
      Array<int> test_vdofs2;
      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         FaceElementTransformations *ftr = mesh->GetFaceElementTransformations(i);
         if (ftr != NULL)
         {
            trial_fes->GetFaceVDofs(i, trial_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
            const FiniteElement *trial_face_fe = trial_fes->GetFaceElement(i);
            const FiniteElement *test_fe1 = test_fes->GetFE(ftr->Elem1No);
            const FiniteElement *test_fe2;
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
   }

   if (boundary_trace_face_integs.Size())
   {
      for (int i = 0; i < mesh->GetNBE(); i++)
      {
         int bdr_attr = mesh->GetBdrAttribute(i);
         FaceElementTransformations *ftr = mesh->GetBdrFaceTransformations(i);
         if (ftr != NULL)
         {
            trial_fes->GetFaceVDofs(ftr->ElementNo, trial_vdofs);
            test_fes->GetElementVDofs(ftr->Elem1No, test_vdofs);
            const FiniteElement *trial_face_fe = trial_fes->GetFaceElement(ftr->ElementNo);
            const FiniteElement *test_fe1 = test_fes->GetFE(ftr->Elem1No);
            // The test_fe2 object is really a dummy and not used on the
            // boundaries, but we can't dereference a NULL pointer, and we don't
            // want to actually make a fake element.
            const FiniteElement *test_fe2 = test_fe1;
            for (int k = 0; k < boundary_trace_face_integs.Size(); k++)
            {
               if (boundary_trace_face_integs_marker[k] == NULL ||
                   (*boundary_trace_face_integs_marker[k])[bdr_attr-1] == 1)
               {
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
}

void MixedBilinearForm::ConformingAssemble()
{
   if (assembly != AssemblyLevel::LEGACY)
   {
      MFEM_WARNING("Conforming assemble not supported for this assembly level!");
      return;
   }

   const int remove_zeros = 0;
   Finalize(remove_zeros);

   const SparseMatrix *test_P = test_fes->GetConformingProlongation();
   if (test_P)
   {
      SparseMatrix *RA = mfem::TransposeMult(*test_P, *mat);
      delete mat;
      mat = RA;
   }

   const SparseMatrix *trial_P = trial_fes->GetConformingProlongation();
   if (trial_P)
   {
      SparseMatrix *RAP = mfem::Mult(*mat, *trial_P);
      delete mat;
      mat = RAP;
   }

   height = mat->Height();
   width = mat->Width();
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

void MixedBilinearForm::FormRectangularLinearSystem(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   Vector &x, Vector &b,
   OperatorHandle &A,
   Vector &X, Vector &B)
{
   if (ext)
   {
      Operator *oper;
      ext->FormRectangularLinearSystem(trial_tdof_list, test_tdof_list,
                                       x, b, oper, X, B);
      A.Reset(oper);
      return;
   }

   const Operator *Pi = this->GetProlongation();
   const Operator *Po = this->GetOutputProlongation();
   const Operator *Ri = this->GetRestriction();
   InitTVectors(Po, Ri, Pi, x, b, X, B);

   if (!mat_e)
   {
      // Set A = mat_e
      FormRectangularSystemMatrix(trial_tdof_list, test_tdof_list, A);
   }
   // Eliminate essential BCs with B -= Ab xb
   mat_e->AddMult(X, B, -1.0);

   B.SetSubVector(test_tdof_list, 0.0);
}

void MixedBilinearForm::FormRectangularSystemMatrix(
   const Array<int> &trial_tdof_list,
   const Array<int> &test_tdof_list,
   OperatorHandle &A)
{
   if (ext)
   {
      Operator *oper;
      ext->FormRectangularSystemOperator(trial_tdof_list, test_tdof_list, oper);
      A.Reset(oper);
      return;
   }

   ConformingAssemble();

   Array<int> ess_trial_tdof_marker, ess_test_tdof_marker;
   FiniteElementSpace::ListToMarker(trial_tdof_list, trial_fes->GetTrueVSize(),
                                    ess_trial_tdof_marker);
   FiniteElementSpace::ListToMarker(test_tdof_list, test_fes->GetTrueVSize(),
                                    ess_test_tdof_marker);

   mat_e = new SparseMatrix(mat->Height(), mat->Width());
   mat->EliminateCols(ess_trial_tdof_marker, *mat_e);

   for (int i = 0; i < test_tdof_list.Size(); i++)
   {
      mat->EliminateRow(test_tdof_list[i]);
   }
   mat_e->Finalize();
   A.Reset(mat, false);
}

void MixedBilinearForm::EliminateTrialDofs(
   const Array<int> &bdr_attr_is_ess, const Vector &sol, Vector &rhs)
{
   int i, j, k;
   Array<int> ess_dofs, cols_marker(trial_fes->GetVSize());
   cols_marker = 0;

   for (i = 0; i < trial_fes->GetNBE(); i++)
   {
      if (bdr_attr_is_ess[trial_fes->GetBdrAttribute(i)-1])
      {
         trial_fes->GetBdrElementVDofs(i, ess_dofs);
         for (j = 0; j < ess_dofs.Size(); j++)
         {
            if ((k = ess_dofs[j]) < 0)
            {
               k = -1-k;
            }
            cols_marker[k] = 1;
         }
      }
   }
   mat->EliminateCols(cols_marker, &sol, &rhs);
}

void MixedBilinearForm::EliminateEssentialBCFromTrialDofs(
   const Array<int> &marked_vdofs, const Vector &sol, Vector &rhs)
{
   mat->EliminateCols(marked_vdofs, &sol, &rhs);
}

void MixedBilinearForm::EliminateTestDofs(const Array<int> &bdr_attr_is_ess)
{
   int i, j, k;
   Array<int> ess_dofs;

   for (i = 0; i < test_fes->GetNBE(); i++)
   {
      if (bdr_attr_is_ess[test_fes->GetBdrAttribute(i)-1])
      {
         test_fes->GetBdrElementVDofs(i, ess_dofs);
         for (j = 0; j < ess_dofs.Size(); j++)
         {
            if ((k = ess_dofs[j]) < 0)
            {
               k = -1-k;
            }
            mat->EliminateRow(k);
         }
      }
   }
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

   Mesh *mesh = test_fes->GetMesh();
   DenseMatrix elmat;
   Array<int> trial_vdofs, test_vdofs;

   if (mat == NULL)
   {
      mat = new SparseMatrix(height, width);
   }

   if (domain_integs.Size())
   {
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         DofTransformation *dom_dof_trans = trial_fes->GetElementVDofs(i, trial_vdofs);
         DofTransformation *ran_dof_trans = test_fes->GetElementVDofs(i, test_vdofs);
         ElementTransformation *eltrans = test_fes->GetElementTransformation(i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int j = 0; j < domain_integs.Size(); j++)
         {
            domain_integs[j]->AssembleElementMatrix2(*trial_fes->GetFE(i),
                                                     *test_fes->GetFE(i),
                                                     *eltrans, elemmat);
            elmat += elemmat;
         }
         if (ran_dof_trans || dom_dof_trans)
         {
            TransformPrimal(ran_dof_trans, dom_dof_trans, elmat);
         }
         mat->SetSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
      }
   }

   if (trace_face_integs.Size())
   {
      for (int i = 0; i < mesh->GetNumFaces(); i++)
      {
         trial_fes->GetFaceVDofs(i, trial_vdofs);
         test_fes->GetFaceVDofs(i, test_vdofs);
         ElementTransformation *eltrans = mesh->GetFaceTransformation(i);

         elmat.SetSize(test_vdofs.Size(), trial_vdofs.Size());
         elmat = 0.0;
         for (int j = 0; j < trace_face_integs.Size(); j++)
         {
            trace_face_integs[j]->AssembleElementMatrix2(*trial_fes->GetFaceElement(i),
                                                         *test_fes->GetFaceElement(i),
                                                         *eltrans, elemmat);
            elmat += elemmat;
         }
         mat->SetSubMatrix(test_vdofs, trial_vdofs, elmat, skip_zeros);
      }
   }
}

void DiscreteLinearOperator::FormDiscreteOperatorMatrix(OperatorHandle &A)
{
   if (ext)
   {
      Operator *oper;
      ext->FormDiscreteOperator(oper);
      A.Reset(oper);
      return;
   }

   mat->Finalize();

   const SparseMatrix *test_R = test_fes->GetConformingRestriction();
   if (test_R)
   {
      SparseMatrix *RA = mfem::Mult(*test_R, *mat);
      delete mat;
      mat = RA;
   }

   const SparseMatrix *trial_P = trial_fes->GetConformingProlongation();
   if (trial_P)
   {
      SparseMatrix *RAP = mfem::Mult(*mat, *trial_P);
      delete mat;
      mat = RAP;
   }

   height = mat->Height();
   width = mat->Width();

   A.Reset(mat, false);
}

} // namespace mfem
