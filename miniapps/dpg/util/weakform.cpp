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

#include "weakform.hpp"

namespace mfem
{

void DPGWeakForm::Init()
{
   trial_integs.SetSize(trial_fes.Size(), test_fecols.Size());
   for (int i = 0; i < trial_integs.NumRows(); i++)
   {
      for (int j = 0; j < trial_integs.NumCols(); j++)
      {
         trial_integs(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   test_integs.SetSize(test_fecols.Size(), test_fecols.Size());
   for (int i = 0; i < test_integs.NumRows(); i++)
   {
      for (int j = 0; j < test_integs.NumCols(); j++)
      {
         test_integs(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   lfis.SetSize(test_fecols.Size());
   for (int j = 0; j < lfis.Size(); j++)
   {
      lfis[j] = new Array<LinearFormIntegrator * >();
   }


   ComputeOffsets();

   mat = mat_e = NULL;
   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;

   initialized = true;
   static_cond = nullptr;

   if (store_matrices)
   {
      Bmat.SetSize(mesh->GetNE());
      fvec.SetSize(mesh->GetNE());
   }
}

void DPGWeakForm::ComputeOffsets()
{
   dof_offsets.SetSize(nblocks+1);
   tdof_offsets.SetSize(nblocks+1);
   dof_offsets[0] = 0;
   tdof_offsets[0] = 0;
   for (int i =0; i<nblocks; i++)
   {
      dof_offsets[i+1] = trial_fes[i]->GetVSize();
      tdof_offsets[i+1] = trial_fes[i]->GetTrueVSize();
   }
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
}

// Allocate SparseMatrix and RHS
void DPGWeakForm::AllocMat()
{
   if (static_cond) { return; }

   mat = new BlockMatrix(dof_offsets);
   mat->owns_blocks = 1;

   for (int i = 0; i<mat->NumRowBlocks(); i++)
   {
      int h = dof_offsets[i+1] - dof_offsets[i];
      for (int j = 0; j<mat->NumColBlocks(); j++)
      {
         int w = dof_offsets[j+1] - dof_offsets[j];
         mat->SetBlock(i,j,new SparseMatrix(h, w));
      }
   }
   y = new BlockVector(dof_offsets);
   *y = 0.;
}

void DPGWeakForm::Finalize(int skip_zeros)
{
   if (mat) { mat->Finalize(skip_zeros); }
   if (mat_e) { mat_e->Finalize(skip_zeros); }
   if (static_cond) { static_cond->Finalize(); }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void DPGWeakForm::AddTrialIntegrator(
   BilinearFormIntegrator *bfi, int n, int m)
{
   MFEM_VERIFY(n>=0 && n<trial_fes.Size(),
               "DPGWeakFrom::AddTrialIntegrator: trial fespace index out of bounds");
   MFEM_VERIFY(m>=0 && m<test_fecols.Size(),
               "DPGWeakFrom::AddTrialIntegrator: test fecol index out of bounds");
   trial_integs(n,m)->Append(bfi);
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void DPGWeakForm::AddTestIntegrator
(BilinearFormIntegrator *bfi, int n, int m)
{
   MFEM_VERIFY(n>=0 && n<test_fecols.Size() && m>=0 && m<test_fecols.Size(),
               "DPGWeakFrom::AdTestIntegrator: test fecol index out of bounds");
   test_integs(n,m)->Append(bfi);
}

/// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
void DPGWeakForm::AddDomainLFIntegrator(
   LinearFormIntegrator *lfi, int n)
{
   MFEM_VERIFY(n>=0 && n<test_fecols.Size(),
               "DPGWeakFrom::AddDomainLFIntegrator: test fecol index out of bounds");
   lfis[n]->Append(lfi);
}

void DPGWeakForm::BuildProlongation()
{
   P = new BlockMatrix(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      const SparseMatrix *P_ = trial_fes[i]->GetConformingProlongation();
      if (P_)
      {
         const SparseMatrix *R_ = trial_fes[i]->GetRestrictionMatrix();
         P->SetBlock(i,i,const_cast<SparseMatrix*>(P_));
         R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
      }
   }
}

void DPGWeakForm::ConformingAssemble()
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA = mfem::Mult(*Pt, *mat);
   mat->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      for (int j = 0; j<nblocks; j++)
      {
         SparseMatrix * tmp = &mat->GetBlock(i,j);
         if (Pt->IsZeroBlock(i,i))
         {
            PtA->SetBlock(i,j,tmp);
         }
         else
         {
            delete tmp;
         }
      }
   }
   delete mat;
   if (mat_e)
   {
      BlockMatrix *PtAe = mfem::Mult(*Pt, *mat_e);
      mat_e->owns_blocks = 0;
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            SparseMatrix * tmp = &mat_e->GetBlock(i,j);
            if (Pt->IsZeroBlock(i,i))
            {
               PtAe->SetBlock(i,j,tmp);
            }
            else
            {
               delete tmp;
            }
         }
      }
      delete mat_e;
      mat_e = PtAe;
   }
   delete Pt;

   mat = mfem::Mult(*PtA, *P);

   PtA->owns_blocks = 0;
   for (int i = 0; i<nblocks; i++)
   {
      for (int j = 0; j<nblocks; j++)
      {
         SparseMatrix * tmp = &PtA->GetBlock(j,i);
         if (P->IsZeroBlock(i,i))
         {
            mat->SetBlock(j,i,tmp);
         }
         else
         {
            delete tmp;
         }
      }
   }
   delete PtA;

   if (mat_e)
   {
      BlockMatrix *PtAeP = mfem::Mult(*mat_e, *P);
      mat_e->owns_blocks = 0;
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            SparseMatrix * tmp = &mat_e->GetBlock(j,i);
            if (P->IsZeroBlock(i,i))
            {
               PtAeP->SetBlock(j,i,tmp);
            }
            else
            {
               delete tmp;
            }
         }
      }

      delete mat_e;
      mat_e = PtAeP;
   }
   height = mat->Height();
   width = mat->Width();
}

/// Assembles the form i.e. sums over all domain integrators.
void DPGWeakForm::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   Array<int> faces, ori;

   DofTransformation * doftrans_i, *doftrans_j;
   if (mat == NULL)
   {
      AllocMat();
   }

   // loop through the elements
   int dim = mesh->Dimension();
   DenseMatrix B, Be, G, Ge, A;
   Vector vec_e, vec, Gvec, b;
   Array<int> vdofs;

   // loop through elements
   for (int iel = 0; iel < mesh -> GetNE(); iel++)
   {
      if (dim == 1)
      {
         mesh->GetElementVertices(iel, faces);
      }
      else if (dim == 2)
      {
         mesh->GetElementEdges(iel, faces, ori);
      }
      else if (dim == 3)
      {
         mesh->GetElementFaces(iel,faces,ori);
      }
      else
      {
         MFEM_ABORT("DPGWeakForm::Assemble: dim > 3 not supported");
      }
      int numfaces = faces.Size();

      Array<int> test_offs(test_fecols.Size()+1); test_offs[0] = 0;
      Array<int> trial_offs(trial_fes.Size()+1); trial_offs = 0;

      eltrans = mesh->GetElementTransformation(iel);
      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder(); // assuming uniform order
         test_offs[j+1] = test_fecols_vdims[j]*test_fecols[j]->GetFE(
                             eltrans->GetGeometryType(),
                             order)->GetDof();
      }
      for (int j = 0; j < trial_fes.Size(); j++)
      {
         if (IsTraceFes[j])
         {
            for (int ie = 0; ie<faces.Size(); ie++)
            {
               trial_offs[j+1] += trial_fes[j]->GetVDim()*trial_fes[j]->GetFaceElement(
                                     faces[ie])->GetDof();
            }
         }
         else
         {
            trial_offs[j+1] = trial_fes[j]->GetVDim() * trial_fes[j]->GetFE(
                                 iel)->GetDof();
         }
      }
      test_offs.PartialSum();
      trial_offs.PartialSum();

      G.SetSize(test_offs.Last()); G = 0.0;
      vec.SetSize(test_offs.Last()); vec = 0.0;
      B.SetSize(test_offs.Last(),trial_offs.Last()); B = 0.0;


      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order_j = test_fecols[j]->GetOrder();

         eltrans = mesh->GetElementTransformation(iel);
         const FiniteElement & test_fe =
            *test_fecols[j]->GetFE(eltrans->GetGeometryType(), order_j);

         for (int k = 0; k < lfis[j]->Size(); k++)
         {
            (*lfis[j])[k]->AssembleRHSElementVect(test_fe,*eltrans,vec_e);
            vec.AddSubVector(vec_e,test_offs[j]);
         }

         for (int i = 0; i < test_fecols.Size(); i++)
         {
            int order_i = test_fecols[i]->GetOrder();
            eltrans = mesh->GetElementTransformation(iel);
            const FiniteElement & test_fe_i =
               *test_fecols[i]->GetFE(eltrans->GetGeometryType(), order_i);

            for (int k = 0; k < test_integs(i,j)->Size(); k++)
            {
               if (i==j)
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix(test_fe,*eltrans,Ge);
               }
               else
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix2(test_fe_i,test_fe,*eltrans,
                                                                 Ge);
               }
               G.AddSubMatrix(test_offs[j], test_offs[i], Ge);
            }
         }

         for (int i = 0; i < trial_fes.Size(); i++)
         {
            if (IsTraceFes[i])
            {
               for (int k = 0; k < trial_integs(i,j)->Size(); k++)
               {
                  int face_dof_offs = 0;
                  for (int ie = 0; ie < numfaces; ie++)
                  {
                     int iface = faces[ie];
                     FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
                     const FiniteElement & tfe = *trial_fes[i]->GetFaceElement(iface);
                     (*trial_integs(i,j))[k]->AssembleTraceFaceMatrix(iel,tfe,test_fe,*ftr,Be);
                     B.AddSubMatrix(test_offs[j], trial_offs[i]+face_dof_offs, Be);
                     face_dof_offs+=Be.Width();
                  }
               }
            }
            else
            {
               const FiniteElement & fe = *trial_fes[i]->GetFE(iel);
               eltrans = mesh->GetElementTransformation(iel);
               for (int k = 0; k < trial_integs(i,j)->Size(); k++)
               {
                  (*trial_integs(i,j))[k]->AssembleElementMatrix2(fe,test_fe,*eltrans,Be);
                  B.AddSubMatrix(test_offs[j], trial_offs[i], Be);
               }
            }
         }
      }

      // Form Normal Equations B^T G^-1 B = B^T G^-1 l
      Gvec.SetSize(G.Height());
      b.SetSize(B.Width());
      A.SetSize(B.Width());

      CholeskyFactors chol(G.GetData());
      chol.Factor(G.Height());

      chol.LSolve(B.Height(), B.Width(), B.GetData());
      chol.LSolve(vec.Size(), 1, vec.GetData());
      if (store_matrices)
      {
         Bmat[iel] = new DenseMatrix(B);
         fvec[iel] = new Vector(vec);
      }
      mfem::MultAtB(B,B,A);
      B.MultTranspose(vec,b);

      if (static_cond)
      {
         static_cond->AssembleReducedSystem(iel,A,b);
      }
      else
      {
         // Assembly
         for (int i = 0; i<trial_fes.Size(); i++)
         {
            Array<int> vdofs_i;
            doftrans_i = nullptr;
            if (IsTraceFes[i])
            {
               Array<int> face_vdofs;
               for (int k = 0; k < numfaces; k++)
               {
                  int iface = faces[k];
                  trial_fes[i]->GetFaceVDofs(iface, face_vdofs);
                  vdofs_i.Append(face_vdofs);
               }
            }
            else
            {
               doftrans_i = trial_fes[i]->GetElementVDofs(iel, vdofs_i);
            }
            for (int j = 0; j<trial_fes.Size(); j++)
            {
               Array<int> vdofs_j;
               doftrans_j = nullptr;

               if (IsTraceFes[j])
               {
                  Array<int> face_vdofs;
                  for (int k = 0; k < numfaces; k++)
                  {
                     int iface = faces[k];
                     trial_fes[j]->GetFaceVDofs(iface, face_vdofs);
                     vdofs_j.Append(face_vdofs);
                  }
               }
               else
               {
                  doftrans_j = trial_fes[j]->GetElementVDofs(iel, vdofs_j);
               }

               DenseMatrix Ae;
               A.GetSubMatrix(trial_offs[i],trial_offs[i+1],
                              trial_offs[j],trial_offs[j+1], Ae);
               if (doftrans_i || doftrans_j)
               {
                  TransformDual(doftrans_i, doftrans_j, Ae);
               }
               mat->GetBlock(i,j).AddSubMatrix(vdofs_i,vdofs_j, Ae);
            }

            // assemble rhs
            double * data = b.GetData();
            Vector vec1;
            // ref subvector
            vec1.SetDataAndSize(&data[trial_offs[i]],
                                trial_offs[i+1]-trial_offs[i]);
            if (doftrans_i)
            {
               doftrans_i->TransformDual(vec1);
            }
            y->GetBlock(i).AddElementVector(vdofs_i,vec1);
         }
      }
   }
}

void DPGWeakForm::FormLinearSystem(const Array<int>
                                   &ess_tdof_list,
                                   Vector &x,
                                   OperatorHandle &A, Vector &X,
                                   Vector &B, int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   if (static_cond)
   {
      // Schur complement reduction to the exposed dofs
      static_cond->ReduceSystem(x, X, B, copy_interior);
   }
   else if (!P)
   {
      EliminateVDofsInRHS(ess_tdof_list, x, *y);
      X.MakeRef(x, 0, x.Size());
      B.MakeRef(*y, 0, y->Size());
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
   else // non conforming space
   {
      B.SetSize(P->Width());

      P->MultTranspose(*y, B);
      double *data = y->GetData();
      Vector tmp;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            B.SetVector(tmp,offset);
         }
      }

      X.SetSize(R->Height());

      R->Mult(x, X);
      data = x.GetData();
      for (int i = 0; i<nblocks; i++)
      {
         if (R->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            X.SetVector(tmp,offset);
         }
      }

      EliminateVDofsInRHS(ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
}

void DPGWeakForm::FormSystemMatrix(const Array<int>
                                   &ess_tdof_list,
                                   OperatorHandle &A)
{
   if (static_cond)
   {
      if (!static_cond->HasEliminatedBC())
      {
         static_cond->SetEssentialTrueDofs(ess_tdof_list);
         static_cond->FormSystemMatrix(diag_policy);
      }
      A.Reset(&static_cond->GetSchurMatrix(), false);
   }
   else
   {
      if (!mat_e)
      {
         bool conforming = true;
         for (int i = 0; i<nblocks; i++)
         {
            const SparseMatrix *P_ = trial_fes[i]->GetConformingProlongation();
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
      A.Reset(mat, false);
   }
}

void DPGWeakForm::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x,b,-1.);
   mat->PartMult(vdofs,x,b);
}

void DPGWeakForm::EliminateVDofs(const Array<int> &vdofs,
                                 Operator::DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      Array<int> offsets;

      offsets.MakeRef( (P) ? tdof_offsets : dof_offsets);

      mat_e = new BlockMatrix(offsets);
      mat_e->owns_blocks = 1;
      for (int i = 0; i<mat_e->NumRowBlocks(); i++)
      {
         int h = offsets[i+1] - offsets[i];
         for (int j = 0; j<mat_e->NumColBlocks(); j++)
         {
            int w = offsets[j+1] - offsets[j];
            mat_e->SetBlock(i,j,new SparseMatrix(h, w));
         }
      }
   }
   mat->EliminateRowCols(vdofs,mat_e,diag_policy);
}

void DPGWeakForm::RecoverFEMSolution(const Vector &X,
                                     Vector &x)
{

   if (static_cond)
   {
      // Private dofs back solve
      static_cond->ComputeSolution(X, x);
   }
   else if (!P)
   {
      x.SyncMemory(X);
   }
   else
   {
      x.SetSize(P->Height());
      P->Mult(X, x);
      double *data = X.GetData();
      Vector tmp;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp.SetDataAndSize(&data[offset],tdof_offsets[i+1]-tdof_offsets[i]);
            x.SetVector(tmp,offset);
         }
      }
   }
}

void DPGWeakForm::ReleaseInitMemory()
{
   if (initialized)
   {
      for (int k = 0; k< trial_integs.NumRows(); k++)
      {
         for (int l = 0; l<trial_integs.NumCols(); l++)
         {
            for (int i = 0; i<trial_integs(k,l)->Size(); i++)
            {
               delete (*trial_integs(k,l))[i];
            }
            delete trial_integs(k,l);
         }
      }
      trial_integs.DeleteAll();

      for (int k = 0; k < test_integs.NumRows(); k++)
      {
         for (int l = 0; l < test_integs.NumCols(); l++)
         {
            for (int i = 0; i < test_integs(k,l)->Size(); i++)
            {
               delete (*test_integs(k,l))[i];
            }
            delete test_integs(k,l);
         }
      }
      test_integs.DeleteAll();

      for (int k = 0; k < lfis.Size(); k++)
      {
         for (int i = 0; i < lfis[k]->Size(); i++)
         {
            delete (*lfis[k])[i];
         }
         delete lfis[k];
      }
      lfis.DeleteAll();
   }
}

void DPGWeakForm::Update()
{
   delete mat_e; mat_e = nullptr;
   delete mat; mat = nullptr;
   delete y; y = nullptr;

   if (P)
   {
      delete P; P = nullptr;
      delete R; R = nullptr;
   }

   if (static_cond)
   {
      EnableStaticCondensation();
   }
   else
   {
      delete static_cond; static_cond = nullptr;
   }

   ComputeOffsets();

   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;

   initialized = true;

   if (store_matrices)
   {
      for (int i = 0; i<Bmat.Size(); i++)
      {
         delete Bmat[i]; Bmat[i] = nullptr;
         delete fvec[i]; fvec[i] = nullptr;
      }
      Bmat.SetSize(mesh->GetNE());
      fvec.SetSize(mesh->GetNE());
      for (int i = 0; i<Bmat.Size(); i++)
      {
         Bmat[i] = nullptr;
         fvec[i] = nullptr;
      }
   }
}

void DPGWeakForm::EnableStaticCondensation()
{
   delete static_cond;
   static_cond = new BlockStaticCondensation(trial_fes);
}

Vector & DPGWeakForm::ComputeResidual(const BlockVector & x)
{
   // Element vector of trial space size
   Vector u;
   Array<int> vdofs;
   Array<int> faces, ori;
   int dim = mesh->Dimension();
   residuals.SetSize(mesh->GetNE());
   // loop through elements
   for (int iel = 0; iel < mesh -> GetNE(); iel++)
   {
      if (dim == 1)
      {
         mesh->GetElementVertices(iel, faces);
      }
      else if (dim == 2)
      {
         mesh->GetElementEdges(iel, faces, ori);
      }
      else if (dim == 3)
      {
         mesh->GetElementFaces(iel,faces,ori);
      }
      else
      {
         MFEM_ABORT("DPGWeakForm::ComputeResidual: "
                    "dim > 3 not supported");
      }
      int numfaces = faces.Size();

      Array<int> trial_offs(trial_fes.Size()+1); trial_offs = 0;

      for (int j = 0; j < trial_fes.Size(); j++)
      {
         if (IsTraceFes[j])
         {
            for (int ie = 0; ie<faces.Size(); ie++)
            {
               trial_offs[j+1] += trial_fes[j]->GetFaceElement(faces[ie])->GetDof();
            }
         }
         else
         {
            trial_offs[j+1] = trial_fes[j]->GetVDim() * trial_fes[j]->GetFE(
                                 iel)->GetDof();
         }
      }
      trial_offs.PartialSum();

      u.SetSize(trial_offs.Last());
      double * data = u.GetData();
      DofTransformation * doftrans = nullptr;
      for (int i = 0; i<trial_fes.Size(); i++)
      {
         vdofs.SetSize(0);
         doftrans = nullptr;
         if (IsTraceFes[i])
         {
            Array<int> face_vdofs;
            for (int k = 0; k < numfaces; k++)
            {
               int iface = faces[k];
               trial_fes[i]->GetFaceVDofs(iface, face_vdofs);
               vdofs.Append(face_vdofs);
            }
         }
         else
         {
            doftrans = trial_fes[i]->GetElementVDofs(iel, vdofs);
         }
         Vector vec1;
         vec1.SetDataAndSize(&data[trial_offs[i]],
                             trial_offs[i+1]-trial_offs[i]);
         x.GetBlock(i).GetSubVector(vdofs,vec1);
         if (doftrans)
         {
            doftrans->InvTransformPrimal(vec1);
         }
      } // end of loop through trial spaces

      Vector v(Bmat[iel]->Height());
      Bmat[iel]->Mult(u,v);
      v -= *fvec[iel];
      residuals[iel] = v.Norml2();
   } // end of loop through elements
   return residuals;
}

DPGWeakForm::~DPGWeakForm()
{
   delete mat_e; mat_e = nullptr;
   delete mat; mat = nullptr;
   delete y; y = nullptr;

   ReleaseInitMemory();

   if (P)
   {
      delete P;
      delete R;
   }

   delete static_cond;

   if (store_matrices)
   {
      for (int i = 0; i<mesh->GetNE(); i++)
      {
         delete Bmat[i]; Bmat[i] = nullptr;
         delete fvec[i]; fvec[i] = nullptr;
      }
   }
}

} // namespace mfem
