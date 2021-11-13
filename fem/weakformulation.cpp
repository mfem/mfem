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
   LinearFormIntegrator *bfi)
{
   domain_lf_integs = bfi;
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
   DofTransformation * doftrans_j, *doftrans_k;
   DenseMatrix elmat, *elmat_p;
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
   DenseMatrix B, G;
   Array<DenseMatrix *> Bhat;
   for (int i = 0; i < mesh -> GetNE(); i++)
   {
      // get element matrices associated with the domain_integrator
      elmat.SetSize(0);
      eltrans = mesh->GetElementTransformation(i);
      const FiniteElement & fe = *fes->GetFE(i);
      int order = test_fecol->GetOrder();
      const FiniteElement & test_fe = *test_fecol->GetFE(fe.GetGeomType(), order);

      //Gram Matrix
      test_integ->AssembleElementMatrix(test_fe, *eltrans,G);
      int h = G.Height();

      // Element Matrix B
      domain_bf_integ->AssembleElementMatrix2(fe,test_fe,*eltrans,B);
      MFEM_VERIFY(B.Height() == h, "Check B height");
      int w = B.Width();

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
      Bhat.SetSize(numfaces);
      for (int j = 0; j < numfaces; j++)
      {
         int iface = faces[j];
         FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
         const FiniteElement & fe = *trace_fes->GetFaceElement(j);
         Bhat[j] = new DenseMatrix();
         trace_integ->AssembleTraceFaceMatrix(i,fe,test_fe,*ftr,*Bhat[j]);
         w += Bhat[j]->Width();
      }

      // Size of Global B;
      DenseMatrix BG(h,w);
      BG.SetSubMatrix(0,0,B);
      int jbeg = B.Width();
      // stack the matrices into [B,Bhat]
      for (int k = 0; k<numfaces; k++)
      {
         BG.SetSubMatrix(0,jbeg,*Bhat[k]);
         jbeg+=Bhat[k]->Width();
      }

      // TODO
      // (1) Integrate Linear form l
      // (2) Form Normal Equations B^T G^-1 B, B^T G^-1 l
      // (3) Assemble Matrix and load vector

   }

}



// void NormalEquationsWeakFormulation::FormLinearSystem(const Array<int> &ess_tdof_list,
//                                          Vector &x,
//                                          Vector &b, OperatorHandle &A, Vector &X,
//                                          Vector &B, int copy_interior)
// {
//    FormSystemMatrix(ess_tdof_list, A);

//    if (!P)
//    {
//       EliminateVDofsInRHS(ess_tdof_list, x, b);
//       X.MakeRef(x, 0, x.Size());
//       B.MakeRef(b, 0, b.Size());
//       if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
//    }
//    else // non conforming space
//    {
//       B.SetSize(P->Width());
//       P->MultTranspose(b, B);
//       X.SetSize(R->Height());

//       mfem::out << "R height, width  = " << R->Height() <<" x "<< R->Width() <<
//                 std::endl;

//       R->Mult(x, X);
//       EliminateVDofsInRHS(ess_tdof_list, X, B);
//       if (!copy_interior) { X.SetSubVectorComplement(ess_tdof_list, 0.0); }
//    }

// }

// void NormalEquationsWeakFormulation::FormSystemMatrix(const Array<int> &ess_tdof_list,
//                                          OperatorHandle &A)
// {
//    if (!mat_e)
//    {
//       const SparseMatrix *P_ = fespaces[0]->GetConformingProlongation();
//       if (P_) { ConformingAssemble(); }
//       EliminateVDofs(ess_tdof_list, diag_policy);
//       const int remove_zeros = 0;
//       Finalize(remove_zeros);
//    }
//    A.Reset(mat, false);
// }

// void NormalEquationsWeakFormulation::RecoverFEMSolution(const Vector &X, const Vector &b,
//                                            Vector &x)
// {
//    if (!P)
//    {
//       x.SyncMemory(X);
//    }
//    else
//    {
//       // Apply conforming prolongation
//       x.SetSize(P->Height());
//       P->Mult(X, x);
//    }
// }


// void NormalEquationsWeakFormulation::ComputeElementMatrices()
// {
//    MFEM_ABORT("NormalEquationsWeakFormulation::ComputeElementMatrices:not implemented yet")
// }

// void NormalEquationsWeakFormulation::ComputeElementMatrix(int i, DenseMatrix &elmat)
// {
//    if (element_matrices)
//    {
//       elmat.SetSize(element_matrices->SizeI(), element_matrices->SizeJ());
//       elmat = element_matrices->GetData(i);
//       return;
//    }

//    int nblocks = fespaces.Size();
//    Array<const FiniteElement *> fe(nblocks);
//    ElementTransformation *eltrans;

//    elmat.SetSize(0);
//    if (domain_integs.Size())
//    {
//       for (int j = 0; j<nblocks; j++)
//       {
//          fe[j] = fespaces[j]->GetFE(i);
//       }
//       eltrans = fespaces[0]->GetElementTransformation(i);
//       domain_integs[0]->AssembleElementMatrix(fe, *eltrans, elmat);
//       for (int k = 1; k < domain_integs.Size(); k++)
//       {
//          domain_integs[k]->AssembleElementMatrix(fe, *eltrans, elemmat);
//          elmat += elemmat;
//       }
//    }
//    else
//    {
//       int matsize = 0;
//       for (int j = 0; j<nblocks; j++)
//       {
//          matsize += fespaces[j]->GetFE(i)->GetDof();
//       }
//       elmat.SetSize(matsize);
//       elmat = 0.0;
//    }
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
//                                              const Vector &sol, Vector &rhs,
//                                              DiagonalPolicy dpolicy)
// {
//    MFEM_ABORT("NormalEquationsWeakFormulation::EliminateEssentialBC: not implemented yet");
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBC(const Array<int> &bdr_attr_is_ess,
//                                              DiagonalPolicy dpolicy)
// {
//    MFEM_ABORT("NormalEquationsWeakFormulation::EliminateEssentialBC: not implemented yet");
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBCDiag (const Array<int>
//                                                   &bdr_attr_is_ess,
//                                                   double value)
// {
//    MFEM_ABORT("NormalEquationsWeakFormulation::EliminateEssentialBCDiag: not implemented yet");
// }

// void NormalEquationsWeakFormulation::EliminateVDofs(const Array<int> &vdofs,
//                                        const Vector &sol, Vector &rhs,
//                                        DiagonalPolicy dpolicy)
// {
//    vdofs.HostRead();
//    for (int i = 0; i < vdofs.Size(); i++)
//    {
//       int vdof = vdofs[i];
//       if ( vdof >= 0 )
//       {
//          mat -> EliminateRowCol (vdof, sol(vdof), rhs, dpolicy);
//       }
//       else
//       {
//          mat -> EliminateRowCol (-1-vdof, sol(-1-vdof), rhs, dpolicy);
//       }
//    }
// }

// void NormalEquationsWeakFormulation::EliminateVDofs(const Array<int> &vdofs,
//                                        DiagonalPolicy dpolicy)
// {
//    if (mat_e == NULL)
//    {
//       mat_e = new SparseMatrix(height);
//    }

//    // mat -> EliminateCols(vdofs, *mat_e,)

//    for (int i = 0; i < vdofs.Size(); i++)
//    {
//       int vdof = vdofs[i];
//       if ( vdof >= 0 )
//       {
//          mat -> EliminateRowCol (vdof, *mat_e, dpolicy);
//       }
//       else
//       {
//          mat -> EliminateRowCol (-1-vdof, *mat_e, dpolicy);
//       }
//    }
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBCFromDofs(
//    const Array<int> &ess_dofs, const Vector &sol, Vector &rhs,
//    DiagonalPolicy dpolicy)
// {
//    MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");
//    MFEM_ASSERT(sol.Size() == height, "incorrect sol Vector size");
//    MFEM_ASSERT(rhs.Size() == height, "incorrect rhs Vector size");

//    for (int i = 0; i < ess_dofs.Size(); i++)
//    {
//       if (ess_dofs[i] < 0)
//       {
//          mat -> EliminateRowCol (i, sol(i), rhs, dpolicy);
//       }
//    }
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBCFromDofs (const Array<int>
//                                                       &ess_dofs,
//                                                       DiagonalPolicy dpolicy)
// {
//    MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

//    for (int i = 0; i < ess_dofs.Size(); i++)
//    {
//       if (ess_dofs[i] < 0)
//       {
//          mat -> EliminateRowCol (i, dpolicy);
//       }
//    }
// }

// void NormalEquationsWeakFormulation::EliminateEssentialBCFromDofsDiag (
//    const Array<int> &ess_dofs,
//    double value)
// {
//    MFEM_ASSERT(ess_dofs.Size() == height, "incorrect dof Array size");

//    for (int i = 0; i < ess_dofs.Size(); i++)
//    {
//       if (ess_dofs[i] < 0)
//       {
//          mat -> EliminateRowColDiag (i, value);
//       }
//    }
// }

// void NormalEquationsWeakFormulation::EliminateVDofsInRHS(
//    const Array<int> &vdofs, const Vector &x, Vector &b)
// {
//    mat_e->AddMult(x, b, -1.);
//    mat->PartMult(vdofs, x, b);
// }



// NormalEquationsWeakFormulation::~NormalEquationsWeakFormulation()
// {
//    delete mat_e;
//    delete mat;
//    delete element_matrices;

//    for (int k=0; k < domain_integs.Size(); k++)
//    {
//       delete domain_integs[k];
//    }
//    for (int k=0; k < trace_integs.Size(); k++)
//    {
//       delete trace_integs[k];
//    }
//    delete P;
//    delete R;
// }

NormalEquationsWeakFormulation::~NormalEquationsWeakFormulation()
{

}


} // namespace mfem
