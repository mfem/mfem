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
   domain_integs.SetSize(domain_fes.Size(), test_fecols.Size());
   // Initialize
   for (int i = 0; i < domain_integs.NumRows(); i++)
   {
      for (int j = 0; j < domain_integs.NumCols(); j++)
      {
         domain_integs(i,j) = nullptr;
      }
   }
   trace_integs.SetSize(trace_fes.Size(), test_fecols.Size());
   for (int i = 0; i < trace_integs.NumRows(); i++)
   {
      for (int j = 0; j < trace_integs.NumCols(); j++)
      {
         trace_integs(i,j) = nullptr;
      }
   }
   test_integs.SetSize(test_fecols.Size(), test_fecols.Size());
   for (int i = 0; i < test_integs.NumRows(); i++)
   {
      for (int j = 0; j < test_integs.NumCols(); j++)
      {
         trace_integs(i,j) = nullptr;
      }
   }

   domain_lf_integs.SetSize(test_fecols.Size());
   for (int j = 0; j < test_integs.NumCols(); j++)
   {
      domain_lf_integs[j] = nullptr;
   }


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
void NormalEquationsWeakFormulation::AddDomainBFIntegrator(
   BilinearFormIntegrator *bfi, int trial_fes, int test_fes)
{
   if (!domain_integs(trial_fes,test_fes))
   {
      domain_integs(trial_fes,test_fes) = new Array<BilinearFormIntegrator * >();
   }
   domain_integs(trial_fes,test_fes)->Append(bfi);
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::AddTestIntegrator
(BilinearFormIntegrator *bfi, int test_fes0, int test_fes1)
{
   if (!test_integs(test_fes0,test_fes1))
   {
      test_integs(test_fes0,test_fes1) = new Array<BilinearFormIntegrator * >();
   }
   domain_integs(test_fes0,test_fes1)->Append(bfi);
}

/// Adds new Trance element BF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::AddTraceElementBFIntegrator(
   BilinearFormIntegrator * bfi, int trial_fes, int test_fes)
{
   if (!trace_integs(trial_fes,test_fes))
   {
      trace_integs(trial_fes,test_fes) = new Array<BilinearFormIntegrator * >();
   }
   trace_integs(trial_fes,test_fes)->Append(bfi);
}

/// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
void NormalEquationsWeakFormulation::AddDomainLFIntegrator(
   LinearFormIntegrator *lfi, int test_fes)
{
   if (!domain_lf_integs[test_fes])
   {
      domain_lf_integs[test_fes] = new Array<LinearFormIntegrator * >();
   }
   domain_lf_integs[test_fes]->Append(lfi);
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
   Array<int> faces, ori;

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
   // loop through elements
   for (int iel = 0; iel < mesh -> GetNE(); iel++)
   {
      // element trasformation
      eltrans = mesh->GetElementTransformation(iel);

      // loop through test fe spaces
      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder();
         const FiniteElement & test_fe =
            *test_fecols[j]->GetFE(eltrans->GetGeometryType(), order);

         for (int i = 0; i < test_fecols.Size(); i++)
         {
            int order = test_fecols[i]->GetOrder();
            const FiniteElement & test_fe_i =
               *test_fecols[i]->GetFE(eltrans->GetGeometryType(), order);

            // loop though test_integ for this combination
            if (!test_integs(i,j)) { continue; }
            for (int k = 0; test_integs(i,j)->Size(); k++)
            {
               if (i==j)
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix(test_fe,*eltrans,G);
               }
               else
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix2(test_fe_i,test_fe,*eltrans,G);
               }
            }
            // TODO
            // need to accumulate and store block G appropriately
         } // end of 2nd loop though test spaces

         // loop through domain trial spaces
         for (int i = 0; i < domain_fes.Size(); i++)
         {
            const FiniteElement & fe = *domain_fes[i]->GetFE(iel);
            // loop though domain_integs
            if (!domain_integs(i,j)) { continue; }
            for (int k = 0; domain_integs(i,j)->Size(); k++)
            {
               (*domain_integs(i,j))[k]->AssembleElementMatrix2(fe,test_fe,*eltrans,Bf);
            }
            // TODO
            // need to accumulate and store block Bf appropriately
         } // end of loop through domain trial spaces


         if (dim == 2)
         {
            mesh->GetElementEdges(iel, faces, ori);
         }
         else if (dim == 3)
         {
            mesh->GetElementFaces(iel,faces,ori);
         }
         int numfaces = faces.Size();
         Bh.SetSize(numfaces);
         // loop through trace trial spaces
         for (int i = 0; i < trace_fes.Size(); i++)
         {
            for (int ie = 0; ie < numfaces; ie++)
            {
               int iface = faces[ie];
               FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
               const FiniteElement & tfe = *trace_fes[i]->GetFaceElement(iface);
               Bh[ie] = new DenseMatrix();
               if (!trace_integs(i,j)) { continue; }
               for (int k = 0; trace_integs(i,j)->Size(); k++)
               {
                  (*trace_integs(i,j))[k]->AssembleTraceFaceMatrix(iel,tfe,test_fe,*ftr,*Bh[j]);
               }

            }
            // TODO
            // need to accumulate and store block Bf appropriately
         }  // end of loop through trace trial spaces

         //    // Size of BG;
         //    DenseMatrix B(h,w);
         //    B.SetSubMatrix(0,0,Bf);
         //    int jbeg = Bf.Width();
         //    // stack the matrices into [B,Bhat]
         //    for (int k = 0; k<numfaces; k++)
         //    {
         //       B.SetSubMatrix(0,jbeg,*Bh[k]);
         //       jbeg+=Bh[k]->Width();
         //    }


         //    // TODO
         //    // (1) Integrate Linear form l
         //    Vector l;
         //    domain_lf_integ->AssembleRHSElementVect(test_fe,*eltrans,l);


         //    // (2) Form Normal Equations B^T G^-1 B, B^T G^-1 l
         //    // A = B^T Ginv B
         //    DenseMatrix A;
         //    RAP(G,B,A);

         //    // b = B^T Ginv l
         //    Vector b(A.Height());
         //    Vector Gl(G.Height());
         //    G.Mult(l,Gl);

         //    B.MultTranspose(Gl,b);

         //    // (3) Assemble Matrix and load vector
         //    for (int j = 0; j<nblocks; j++)
         //    {
         //       Array<int> elem_dofs;
         //       DofTransformation * dtrans = fespaces[j]->GetElementVDofs(i,elem_dofs);
         //       if (dtrans)
         //       {
         //          mfem::out<< "DofTrans is not null" << std::endl;
         //          mfem::out<< "j = " << j << std::endl;
         //       }
         //       elementblockoffsets[j+1] = elem_dofs.Size();
         //    }
         //    elementblockoffsets.PartialSum();

         //    vdofs.SetSize(0);

         //    // field dofs;
         //    fespaces[0]->GetElementVDofs(i, vdofs_j);
         //    int offset_j = dof_offsets[0];
         //    offsetvdofs_j.SetSize(vdofs_j.Size());
         //    for (int l = 0; l<vdofs_j.Size(); l++)
         //    {
         //       offsetvdofs_j[l] = vdofs_j[l]<0 ? -offset_j + vdofs_j[l]
         //                          :  offset_j + vdofs_j[l];
         //    }
         //    vdofs.Append(offsetvdofs_j);


         //    // trace dofs;
         //    offset_j = dof_offsets[1];
         //    Array<int> face_vdofs;
         //    for (int j = 0; j < numfaces; j++)
         //    {
         //       int iface = faces[j];
         //       fespaces[1]->GetFaceVDofs(iface, vdofs_j);
         //       face_vdofs.Append(vdofs_j);
         //    }
         //    offsetvdofs_j.SetSize(face_vdofs.Size());
         //    for (int l = 0; l<face_vdofs.Size(); l++)
         //    {
         //       offsetvdofs_j[l] = face_vdofs[l]<0 ? -offset_j + face_vdofs[l]
         //                          :  offset_j + face_vdofs[l];
         //    }
         //    vdofs.Append(offsetvdofs_j);

         //    mat->AddSubMatrix(vdofs,vdofs,A, skip_zeros);
         //    y->AddElementVector(vdofs,b);
      } // end of loop through test spaces
   } // end of loop through elements

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
