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

#include "complexblockform.hpp"

namespace mfem
{

void ComplexBlockForm::Init()
{
   integs_r.SetSize(fes.Size(), fes.Size());
   integs_i.SetSize(fes.Size(), fes.Size());
   for (int i = 0; i < integs_r.NumRows(); i++)
   {
      for (int j = 0; j < integs_r.NumCols(); j++)
      {
         integs_r(i,j) = new Array<BilinearFormIntegrator * >();
         integs_i(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   ComputeOffsets();

   mat_r = mat_e_r = NULL;
   mat_i = mat_e_i = NULL;
   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;

   initialized = true;

}

void ComplexBlockForm::ComputeOffsets()
{
   dof_offsets.SetSize(nblocks+1);
   tdof_offsets.SetSize(nblocks+1);
   dof_offsets[0] = 0;
   tdof_offsets[0] = 0;
   for (int i =0; i<nblocks; i++)
   {
      dof_offsets[i+1] = fes[i]->GetVSize();
      tdof_offsets[i+1] = fes[i]->GetTrueVSize();
   }
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
}

// Allocate SparseMatrix and RHS
void ComplexBlockForm::AllocMat()
{
   mat_r = new BlockMatrix(dof_offsets);
   mat_r->owns_blocks = 1;
   mat_i = new BlockMatrix(dof_offsets);
   mat_i->owns_blocks = 1;

   for (int i = 0; i < mat_r->NumRowBlocks(); i++)
   {
      int h = dof_offsets[i+1] - dof_offsets[i];
      for (int j = 0; j < mat_r->NumColBlocks(); j++)
      {
         int w = dof_offsets[j+1] - dof_offsets[j];
         mat_r->SetBlock(i,j,new SparseMatrix(h, w));
         mat_i->SetBlock(i,j,new SparseMatrix(h, w));
      }
   }
}

void ComplexBlockForm::Finalize(int skip_zeros)
{
   if (mat_r)
   {
      mat_r->Finalize(skip_zeros);
      mat_i->Finalize(skip_zeros);
   }
   if (mat_e_r)
   {
      mat_e_r->Finalize(skip_zeros);
      mat_e_i->Finalize(skip_zeros);
   }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void ComplexBlockForm::AddDomainIntegrator(
   BilinearFormIntegrator *bfi_r,
   BilinearFormIntegrator *bfi_i,
   int n, int m)
{
   MFEM_VERIFY(n < fes.Size(),
               "ComplexBlockFrom::AddDomainIntegrator: fespace row index out of bounds");
   MFEM_VERIFY(m < fes.Size(),
               "ComplexBlockFrom::AddDomainIntegrator: fespace col index out of bounds");
   if (bfi_r) { integs_r(n,m)->Append(bfi_r); }
   if (bfi_i) { integs_i(n,m)->Append(bfi_i); }
}

void ComplexBlockForm::BuildProlongation()
{
   P = new BlockMatrix(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      const SparseMatrix *P_ = fes[i]->GetConformingProlongation();
      if (P_)
      {
         const SparseMatrix *R_ = fes[i]->GetRestrictionMatrix();
         P->SetBlock(i, i, const_cast<SparseMatrix*>(P_));
         R->SetBlock(i, i, const_cast<SparseMatrix*>(R_));
      }
   }
}

void ComplexBlockForm::ConformingAssemble()
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA_r = mfem::Mult(*Pt, *mat_r);
   BlockMatrix * PtA_i = mfem::Mult(*Pt, *mat_i);
   mat_r->owns_blocks = 0;
   mat_i->owns_blocks = 0;
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         SparseMatrix * tmp_r = &mat_r->GetBlock(i,j);
         SparseMatrix * tmp_i = &mat_i->GetBlock(i,j);
         if (Pt->IsZeroBlock(i, i))
         {
            PtA_r->SetBlock(i, j, tmp_r);
            PtA_i->SetBlock(i, j, tmp_i);
         }
         else
         {
            delete tmp_r;
            delete tmp_i;
         }
      }
   }
   delete mat_r;
   delete mat_i;
   if (mat_e_r)
   {
      BlockMatrix *PtAe_r = mfem::Mult(*Pt, *mat_e_r);
      BlockMatrix *PtAe_i = mfem::Mult(*Pt, *mat_e_i);
      mat_e_r->owns_blocks = 0;
      mat_e_i->owns_blocks = 0;
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            SparseMatrix * tmp_r = &mat_e_r->GetBlock(i, j);
            SparseMatrix * tmp_i = &mat_e_i->GetBlock(i, j);
            if (Pt->IsZeroBlock(i, i))
            {
               PtAe_r->SetBlock(i, j, tmp_r);
               PtAe_i->SetBlock(i, j, tmp_i);
            }
            else
            {
               delete tmp_r;
               delete tmp_i;
            }
         }
      }
      delete mat_e_r;
      delete mat_e_i;
      mat_e_r = PtAe_r;
      mat_e_i = PtAe_i;
   }
   delete Pt;

   mat_r = mfem::Mult(*PtA_r, *P);
   mat_i = mfem::Mult(*PtA_i, *P);

   PtA_r->owns_blocks = 0;
   PtA_i->owns_blocks = 0;
   for (int i = 0; i < nblocks; i++)
   {
      for (int j = 0; j < nblocks; j++)
      {
         SparseMatrix * tmp_r = &PtA_r->GetBlock(j, i);
         SparseMatrix * tmp_i = &PtA_i->GetBlock(j, i);
         if (P->IsZeroBlock(i, i))
         {
            mat_r->SetBlock(j, i, tmp_r);
            mat_i->SetBlock(j, i, tmp_i);
         }
         else
         {
            delete tmp_r;
            delete tmp_i;
         }
      }
   }
   delete PtA_r;
   delete PtA_i;

   if (mat_e_r)
   {
      BlockMatrix *PtAeP_r = mfem::Mult(*mat_e_r, *P);
      BlockMatrix *PtAeP_i = mfem::Mult(*mat_e_i, *P);
      mat_e_r->owns_blocks = 0;
      mat_e_i->owns_blocks = 0;
      for (int i = 0; i < nblocks; i++)
      {
         for (int j = 0; j < nblocks; j++)
         {
            SparseMatrix * tmp_r = &mat_e_r->GetBlock(j, i);
            SparseMatrix * tmp_i = &mat_e_i->GetBlock(j, i);
            if (P->IsZeroBlock(i, i))
            {
               PtAeP_r->SetBlock(j, i, tmp_r);
               PtAeP_i->SetBlock(j, i, tmp_i);
            }
            else
            {
               delete tmp_r;
               delete tmp_i;
            }
         }
      }

      delete mat_e_r;
      delete mat_e_i;
      mat_e_r = PtAeP_r;
      mat_e_i = PtAeP_i;
   }
   height = 2*mat_r->Height();
   width = 2*mat_r->Width();
}

/// Assembles the form i.e. sums over all domain integrators.
void ComplexBlockForm::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   Array<int> faces, ori;

   DofTransformation * doftrans_i, *doftrans_j;
   if (mat_r == NULL)
   {
      AllocMat();
   }

   // loop through the elements
   int dim = mesh->Dimension();
   DenseMatrix A_r, Ae_r;
   DenseMatrix A_i, Ae_i;
   Array<int> vdofs;

   // loop through elements
   for (int iel = 0; iel < mesh -> GetNE(); iel++)
   {
      Array<int> offs(fes.Size()+1); offs = 0;

      eltrans = mesh->GetElementTransformation(iel);

      for (int j = 0; j < fes.Size(); j++)
      {
         offs[j+1] = fes[j]->GetVDim() * fes[j]->GetFE(iel)->GetDof();
      }
      offs.PartialSum();

      A_r.SetSize(offs.Last(),offs.Last()); A_r = 0.0;
      A_i.SetSize(offs.Last(),offs.Last()); A_i = 0.0;

      for (int j = 0; j < fes.Size(); j++)
      {
         const FiniteElement & fe_j = *fes[j]->GetFE(iel);
         for (int i = 0; i < fes.Size(); i++)
         {
            const FiniteElement & fe_i = *fes[i]->GetFE(iel);
            // real integrators
            for (int k = 0; k < integs_r(i,j)->Size(); k++)
            {
               if (i == j)
               {
                  (*integs_r(i,j))[k]->AssembleElementMatrix(fe_i,*eltrans,Ae_r);
               }
               else
               {
                  (*integs_r(i,j))[k]->AssembleElementMatrix2(fe_i,fe_j,*eltrans,Ae_r);
               }
               A_r.AddSubMatrix(offs[j], offs[i], Ae_r);
            }
            // imag integrators
            for (int k = 0; k < integs_i(i,j)->Size(); k++)
            {
               if (i == j)
               {
                  (*integs_i(i,j))[k]->AssembleElementMatrix(fe_i,*eltrans,Ae_i);
               }
               else
               {
                  (*integs_i(i,j))[k]->AssembleElementMatrix2(fe_i, fe_j, *eltrans, Ae_i);
               }
               A_i.AddSubMatrix(offs[j], offs[i], Ae_i);
            }
         }
      }

      ComplexDenseMatrix A(&A_r, &A_i, false, false);

      // Assembly
      for (int i = 0; i<fes.Size(); i++)
      {
         Array<int> vdofs_i;
         doftrans_i = fes[i]->GetElementVDofs(iel, vdofs_i);
         for (int j = 0; j < fes.Size(); j++)
         {
            Array<int> vdofs_j;
            doftrans_j = fes[j]->GetElementVDofs(iel, vdofs_j);

            A.real().GetSubMatrix(offs[i],offs[i+1],
                                  offs[j],offs[j+1], Ae_r);
            A.imag().GetSubMatrix(offs[i],offs[i+1],
                                  offs[j],offs[j+1], Ae_i);
            if (doftrans_i || doftrans_j)
            {
               TransformDual(doftrans_i, doftrans_j, Ae_r);
               TransformDual(doftrans_i, doftrans_j, Ae_i);
            }
            mat_r->GetBlock(i,j).AddSubMatrix(vdofs_i,vdofs_j, Ae_r);
            mat_i->GetBlock(i,j).AddSubMatrix(vdofs_i,vdofs_j, Ae_i);
         }
      }
   } // end of loop through elements
}

void ComplexBlockForm::FormLinearSystem(const Array<int>
                                        &ess_tdof_list,
                                        Vector &x,
                                        Vector &b,
                                        OperatorHandle &A,
                                        Vector &X,
                                        Vector &B,
                                        int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);

   Vector x_r(x, 0, x.Size()/2);
   Vector x_i(x, x.Size()/2, x.Size()/2);
   Vector b_r(b, 0, b.Size()/2);
   Vector b_i(b, b.Size()/2, b.Size()/2);
   if (!P)
   {
      EliminateVDofsInRHS(ess_tdof_list, x_r,x_i, b_r, b_i);
      if (!copy_interior)
      {
         x_r.SetSubVectorComplement(ess_tdof_list, 0.0);
         x_i.SetSubVectorComplement(ess_tdof_list, 0.0);
      }
      X.MakeRef(x, 0, x.Size());
      B.MakeRef(b, 0, b.Size());
   }
   else // non conforming space
   {
      B.SetSize(2*P->Width());
      Vector B_r(B, 0, P->Width());
      Vector B_i(B, P->Width(),P->Width());

      P->MultTranspose(b_r, B_r);
      P->MultTranspose(b_i, B_i);
      Vector tmp_r,tmp_i;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp_r.MakeRef(b_r, offset,tdof_offsets[i+1]-tdof_offsets[i]);
            tmp_i.MakeRef(b_i, offset,tdof_offsets[i+1]-tdof_offsets[i]);
            B_r.SetVector(tmp_r,offset);
            B_i.SetVector(tmp_i,offset);
         }
      }

      X.SetSize(2*R->Height());
      Vector X_r(X, 0, X.Size()/2);
      Vector X_i(X, X.Size()/2, X.Size()/2);

      R->Mult(x_r, X_r);
      R->Mult(x_i, X_i);
      for (int i = 0; i<nblocks; i++)
      {
         if (R->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp_r.MakeRef(x_r, offset, tdof_offsets[i+1]-tdof_offsets[i]);
            tmp_i.MakeRef(x_i, offset, tdof_offsets[i+1]-tdof_offsets[i]);
            X_r.SetVector(tmp_r,offset);
            X_i.SetVector(tmp_i,offset);
         }
      }

      EliminateVDofsInRHS(ess_tdof_list, X_r, X_i, B_r, B_i);
      if (!copy_interior)
      {
         X_r.SetSubVectorComplement(ess_tdof_list, 0.0);
         X_i.SetSubVectorComplement(ess_tdof_list, 0.0);
      }
   }
}

void ComplexBlockForm::FormSystemMatrix(const Array<int>
                                        &ess_tdof_list,
                                        OperatorHandle &A)
{
   if (!mat_e_r)
   {
      bool conforming = true;
      for (int i = 0; i<nblocks; i++)
      {
         const SparseMatrix *P_ = fes[i]->GetConformingProlongation();
         if (P_)
         {
            conforming = false;
            break;
         }
      }
      if (!conforming) { ConformingAssemble(); }
      const int remove_zeros = 0;
      EliminateVDofs(ess_tdof_list, diag_policy);
      Finalize(remove_zeros);
   }
   mat = new ComplexOperator(mat_r,mat_i,false,false);
   A.Reset(mat,false);
}

void ComplexBlockForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x_r, const Vector & x_i,
   Vector &b_r, Vector & b_i)
{
   mat_e_r->AddMult(x_r,b_r,-1.);
   mat_e_i->AddMult(x_i,b_r,1.);
   mat_e_r->AddMult(x_i,b_i,-1.);
   mat_e_i->AddMult(x_r,b_i,-1.);
   mat_r->PartMult(vdofs,x_r,b_r);
   mat_r->PartMult(vdofs,x_i,b_i);
}

void ComplexBlockForm::EliminateVDofs(const Array<int> &vdofs,
                                      Operator::DiagonalPolicy dpolicy)
{
   if (mat_e_r == NULL)
   {
      Array<int> offsets;

      offsets.MakeRef( (P) ? tdof_offsets : dof_offsets);

      mat_e_r = new BlockMatrix(offsets);
      mat_e_r->owns_blocks = 1;
      mat_e_i = new BlockMatrix(offsets);
      mat_e_i->owns_blocks = 1;
      for (int i = 0; i < mat_e_r->NumRowBlocks(); i++)
      {
         int h = offsets[i+1] - offsets[i];
         for (int j = 0; j < mat_e_r->NumColBlocks(); j++)
         {
            int w = offsets[j+1] - offsets[j];
            mat_e_r->SetBlock(i, j, new SparseMatrix(h, w));
            mat_e_i->SetBlock(i, j, new SparseMatrix(h, w));
         }
      }
   }
   mat_r->EliminateRowCols(vdofs, mat_e_r, diag_policy);
   mat_i->EliminateRowCols(vdofs, mat_e_i, Operator::DiagonalPolicy::DIAG_ZERO);
}

void ComplexBlockForm::RecoverFEMSolution(const Vector &X, Vector &x)
{
   if (!P)
   {
      x.SyncMemory(X);
   }
   else
   {
      x.SetSize(2*P->Height());
      Vector X_r(const_cast<Vector &>(X), 0, X.Size()/2);
      Vector X_i(const_cast<Vector &>(X), X.Size()/2, X.Size()/2);

      Vector x_r(x, 0, x.Size()/2);
      Vector x_i(x, x.Size()/2, x.Size()/2);

      P->Mult(X_r, x_r);
      P->Mult(X_i, x_i);

      Vector tmp_r, tmp_i;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp_r.MakeRef(X_r, offset, tdof_offsets[i+1]-tdof_offsets[i]);
            tmp_i.MakeRef(X_i, offset, tdof_offsets[i+1]-tdof_offsets[i]);
            x_r.SetVector(tmp_r,offset);
            x_i.SetVector(tmp_i,offset);
         }
      }
   }
}

void ComplexBlockForm::ReleaseInitMemory()
{
   if (initialized)
   {
      for (int k = 0; k < integs_r.NumRows(); k++)
      {
         for (int l = 0; l < integs_r.NumCols(); l++)
         {
            for (int i = 0; i < integs_r(k,l)->Size(); i++)
            {
               delete (*integs_r(k,l))[i];
            }
            delete integs_r(k,l);
            for (int i = 0; i < integs_i(k,l)->Size(); i++)
            {
               delete (*integs_i(k,l))[i];
            }
            delete integs_i(k,l);
         }
      }
      integs_r.DeleteAll();
      integs_i.DeleteAll();
   }
}

void ComplexBlockForm::Update()
{
   delete mat_e_r; mat_e_r = nullptr;
   delete mat_e_i; mat_e_i = nullptr;
   delete mat; mat = nullptr;
   delete mat_r; mat_r = nullptr;
   delete mat_i; mat_i = nullptr;

   if (P)
   {
      delete P; P = nullptr;
      delete R; R = nullptr;
   }

   ComputeOffsets();

   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;

   initialized = true;

}

ComplexBlockForm::~ComplexBlockForm()
{
   delete mat_e_r; mat_e_r = nullptr;
   delete mat_e_i; mat_e_i = nullptr;
   delete mat; mat = nullptr;
   delete mat_r; mat_r = nullptr;
   delete mat_i; mat_i = nullptr;

   ReleaseInitMemory();

   if (P)
   {
      delete P;
      delete R;
   }
}

} // namespace mfem
