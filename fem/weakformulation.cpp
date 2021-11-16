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

#include "fem.hpp"

namespace mfem
{

void NormalEquationsWeakFormulation::Init()
{
   mesh = fes->GetMesh();
   nblocks = fespaces.Size();
   dof_offsets.SetSize(nblocks+1);
   tdof_offsets.SetSize(nblocks+1);
   dof_offsets[0] = 0;
   tdof_offsets[0] = 0;
   for (int i =0; i<nblocks; i++)
   {
      dof_offsets[i+1] = fespaces[i]->GetVSize();
      tdof_offsets[i+1] = fespaces[i]->GetTrueVSize();
   }
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
   mat = mat_e = NULL;
   element_matrices = NULL;
   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;
}


// Allocate appropriate SparseMatrix and assign it to mat
void NormalEquationsWeakFormulation::AllocMat()
{
   mat = new SparseMatrix(height);
   y = new Vector(height);
   *y = 0.;
}

void NormalEquationsWeakFormulation::Finalize(int skip_zeros)
{
   mat->Finalize(skip_zeros);
   if (mat_e) { mat_e->Finalize(skip_zeros); }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::SetDomainBFIntegrator(
   BilinearFormIntegrator *bfi)
{
   domain_bf_integ = bfi;
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::SetTestIntegrator(BilinearFormIntegrator
                                                       *bfi)
{
   test_integ = bfi;
}

/// Adds new Trance element BF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::SetTraceElementBFIntegrator(
   BilinearFormIntegrator * bfi)
{
   trace_integ = bfi;
}

/// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::SetDomainLFIntegrator(
   LinearFormIntegrator *lfi)
{
   domain_lf_integ = lfi;
}

void NormalEquationsWeakFormulation::BuildProlongation()
{
   P = new BlockMatrix(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   for (int i = 0; i<nblocks; i++)
   {
      const SparseMatrix *P_ = fespaces[i]->GetConformingProlongation();
      const SparseMatrix *R_ = fespaces[i]->GetRestrictionMatrix();
      P->SetBlock(i,i,const_cast<SparseMatrix*>(P_));
      R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
   }
}

void NormalEquationsWeakFormulation::ConformingAssemble()
{
   Finalize(0);
   MFEM_ASSERT(mat, "the BilinearForm is not assembled");

   if (!P) { BuildProlongation(); }

   SparseMatrix * Pm = P->CreateMonolithic();

   SparseMatrix *Pt = Transpose(*Pm);

   SparseMatrix *PtA = mfem::Mult(*Pt, *mat);
   delete mat;
   if (mat_e)
   {
      SparseMatrix *PtAe = mfem::Mult(*Pt, *mat_e);
      delete mat_e;
      mat_e = PtAe;
   }
   delete Pt;
   mat = mfem::Mult(*PtA, *Pm);
   delete PtA;
   if (mat_e)
   {
      SparseMatrix *PtAeP = mfem::Mult(*mat_e, *Pm);
      delete mat_e;
      mat_e = PtAeP;
   }
   delete Pm;
   height = mat->Height();
   width = mat->Width();
}


/// Assembles the form i.e. sums over all domain integrators.
void NormalEquationsWeakFormulation::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   // DofTransformation * doftrans_j, *doftrans_k;
   // DenseMatrix elmat, *elmat_p;
   Array<const FiniteElement *> fe(nblocks);
   Array<int> vdofs_j, vdofs_k;
   Array<int> offsetvdofs_j;
   Array<int> elementblockoffsets(nblocks+1);
   elementblockoffsets[0] = 0;
   if (mat == NULL)
   {
      AllocMat();
   }

   // loop through the elements
   int dim = mesh->Dimension();
   DenseMatrix Bf, G;
   Array<DenseMatrix *> Bh;
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      // get element matrices associated with the domain_integrator
      eltrans = mesh->GetElementTransformation(i);
      const FiniteElement & fe = *fes->GetFE(i);
      int order = test_fecol->GetOrder();
      const FiniteElement & test_fe = *test_fecol->GetFE(fe.GetGeomType(), order);

      //Gram Matrix
      test_integ->AssembleElementMatrix(test_fe, *eltrans,G);
      int h = G.Height();

      // Element Matrix B
      domain_bf_integ->AssembleElementMatrix2(fe,test_fe,*eltrans,Bf);
      MFEM_VERIFY(Bf.Height() == h, "Check Bf height");
      int w = Bf.Width();

      // Element Matrix Bhat
      Array<int> faces, ori;
      if (dim == 2)
      {
         mesh->GetElementEdges(i, faces, ori);
      }
      else if (dim == 3)
      {
         mesh->GetElementFaces(i,faces,ori);
      }
      int numfaces = faces.Size();
      Bh.SetSize(numfaces);
      for (int j = 0; j < numfaces; j++)
      {
         int iface = faces[j];
         FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
         const FiniteElement & tfe = *trace_fes->GetFaceElement(iface);
         Bh[j] = new DenseMatrix();
         trace_integ->AssembleTraceFaceMatrix(i,tfe,test_fe,*ftr,*Bh[j]);
         w += Bh[j]->Width();
      }

      // Size of BG;
      DenseMatrix B(h,w);
      B.SetSubMatrix(0,0,Bf);
      int jbeg = Bf.Width();
      // stack the matrices into [B,Bhat]
      for (int k = 0; k<numfaces; k++)
      {
         B.SetSubMatrix(0,jbeg,*Bh[k]);
         jbeg+=Bh[k]->Width();
      }


      // TODO
      // (1) Integrate Linear form l
      Vector l;
      domain_lf_integ->AssembleRHSElementVect(test_fe,*eltrans,l);


      // (2) Form Normal Equations B^T G^-1 B, B^T G^-1 l
      // A = B^T Ginv B
      DenseMatrix A;
      RAP(G,B,A);

      // b = B^T Ginv l
      Vector b(A.Height());
      Vector Gl(G.Height());
      G.Mult(l,Gl);

      B.MultTranspose(Gl,b);

      // (3) Assemble Matrix and load vector
      for (int j = 0; j<nblocks; j++)
      {
         Array<int> elem_dofs;
         DofTransformation * dtrans = fespaces[j]->GetElementVDofs(i,elem_dofs);
         if (dtrans)
         {
            mfem::out<< "DofTrans is not null" << std::endl;
            mfem::out<< "j = " << j << std::endl;
         }
         elementblockoffsets[j+1] = elem_dofs.Size();
      }
      elementblockoffsets.PartialSum();

      vdofs.SetSize(0);

      // field dofs;
      fespaces[0]->GetElementVDofs(i, vdofs_j);
      int offset_j = dof_offsets[0];
      offsetvdofs_j.SetSize(vdofs_j.Size());
      for (int l = 0; l<vdofs_j.Size(); l++)
      {
         offsetvdofs_j[l] = vdofs_j[l]<0 ? -offset_j + vdofs_j[l]
                            :  offset_j + vdofs_j[l];
      }
      vdofs.Append(offsetvdofs_j);


      // trace dofs;
      offset_j = dof_offsets[1];
      Array<int> face_vdofs;
      for (int j = 0; j < numfaces; j++)
      {
         int iface = faces[j];
         fespaces[1]->GetFaceVDofs(iface, vdofs_j);
         face_vdofs.Append(vdofs_j);
      }
      offsetvdofs_j.SetSize(face_vdofs.Size());
      for (int l = 0; l<face_vdofs.Size(); l++)
      {
         offsetvdofs_j[l] = face_vdofs[l]<0 ? -offset_j + face_vdofs[l]
                            :  offset_j + face_vdofs[l];
      }
      vdofs.Append(offsetvdofs_j);

      mat->AddSubMatrix(vdofs,vdofs,A, skip_zeros);
      y->AddElementVector(vdofs,b);
   }

}


void NormalEquationsWeakFormulation::FormLinearSystem(const Array<int>
                                                      &ess_tdof_list,
                                                      Vector &x,
                                                      OperatorHandle &A, Vector &X,
                                                      Vector &B, int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);

   if (!P)
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
      X.SetSize(R->Height());

      R->Mult(x, X);
      EliminateVDofsInRHS(ess_tdof_list, X, B);
      if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
   }
}

void NormalEquationsWeakFormulation::FormSystemMatrix(const Array<int>
                                                      &ess_tdof_list,
                                                      OperatorHandle &A)
{
   if (!mat_e)
   {
      const SparseMatrix *P_ = fespaces[0]->GetConformingProlongation();
      if (P_) { ConformingAssemble(); }
      EliminateVDofs(ess_tdof_list, diag_policy);
      const int remove_zeros = 0;
      Finalize(remove_zeros);
   }
   A.Reset(mat, false);
}

void NormalEquationsWeakFormulation::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
}


void NormalEquationsWeakFormulation::EliminateVDofs(const Array<int> &vdofs,
                                                    Operator::DiagonalPolicy dpolicy)
{
   if (mat_e == NULL)
   {
      mat_e = new SparseMatrix(height);
   }

   // mat -> EliminateCols(vdofs, *mat_e,)

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


void NormalEquationsWeakFormulation::RecoverFEMSolution(const Vector &X,
                                                        Vector &x)
{
   if (!P)
   {
      x.SyncMemory(X);
   }
   else
   {
      // Apply conforming prolongation
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
}

NormalEquationsWeakFormulation::~NormalEquationsWeakFormulation()
{
   // delete mat_e;
   // delete mat;
   // delete element_matrices;

   // for (int k=0; k < domain_integs.Size(); k++)
   // {
   //    delete domain_integs[k];
   // }
   // for (int k=0; k < trace_integs.Size(); k++)
   // {
   //    delete trace_integs[k];
   // }
   // delete P;
   // delete R;
}


} // namespace mfem
