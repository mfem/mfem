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

#include "complexweakform.hpp"

namespace mfem
{

void ComplexDPGWeakForm::Init()
{
   trial_integs_r.SetSize(trial_fes.Size(), test_fecols.Size());
   trial_integs_i.SetSize(trial_fes.Size(), test_fecols.Size());
   for (int i = 0; i < trial_integs_r.NumRows(); i++)
   {
      for (int j = 0; j < trial_integs_r.NumCols(); j++)
      {
         trial_integs_r(i,j) = new Array<BilinearFormIntegrator * >();
         trial_integs_i(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   test_integs_r.SetSize(test_fecols.Size(), test_fecols.Size());
   test_integs_i.SetSize(test_fecols.Size(), test_fecols.Size());
   for (int i = 0; i < test_integs_r.NumRows(); i++)
   {
      for (int j = 0; j < test_integs_r.NumCols(); j++)
      {
         test_integs_r(i,j) = new Array<BilinearFormIntegrator * >();
         test_integs_i(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   lfis_r.SetSize(test_fecols.Size());
   lfis_i.SetSize(test_fecols.Size());
   for (int j = 0; j < lfis_r.Size(); j++)
   {
      lfis_r[j] = new Array<LinearFormIntegrator * >();
      lfis_i[j] = new Array<LinearFormIntegrator * >();
   }

   ComputeOffsets();

   mat_r = mat_e_r = NULL;
   mat_i = mat_e_i = NULL;
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

void ComplexDPGWeakForm::ComputeOffsets()
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
void ComplexDPGWeakForm::AllocMat()
{
   if (static_cond) { return; }

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
   y = new Vector(2*dof_offsets.Last());
   *y=0.;
   y_r = new BlockVector(*y, dof_offsets);
   y_i = new BlockVector(*y, dof_offsets.Last(), dof_offsets);
}

void ComplexDPGWeakForm::Finalize(int skip_zeros)
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
   if (static_cond) { static_cond->Finalize(); }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void ComplexDPGWeakForm::AddTrialIntegrator(
   BilinearFormIntegrator *bfi_r,
   BilinearFormIntegrator *bfi_i,
   int n, int m)
{
   MFEM_VERIFY(n < trial_fes.Size(),
               "ComplexDPGWeakFrom::AddTrialIntegrator: trial fespace index out of bounds");
   MFEM_VERIFY(m < test_fecols.Size(),
               "ComplexDPGWeakFrom::AddTrialIntegrator: test fecol index out of bounds");
   if (bfi_r)
   {
      trial_integs_r(n,m)->Append(bfi_r);
   }
   if (bfi_i)
   {
      trial_integs_i(n,m)->Append(bfi_i);
   }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void ComplexDPGWeakForm::AddTestIntegrator(
   BilinearFormIntegrator *bfi_r,
   BilinearFormIntegrator *bfi_i,
   int n, int m)
{
   MFEM_VERIFY(n < test_fecols.Size() && m < test_fecols.Size(),
               "ComplexDPGWeakFrom::AdTestIntegrator: test fecol index out of bounds");
   if (bfi_r)
   {
      test_integs_r(n,m)->Append(bfi_r);
   }
   if (bfi_i)
   {
      test_integs_i(n,m)->Append(bfi_i);
   }
}

/// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
void ComplexDPGWeakForm::AddDomainLFIntegrator(
   LinearFormIntegrator *lfi_r,
   LinearFormIntegrator *lfi_i, int n)
{
   MFEM_VERIFY(n < test_fecols.Size(),
               "ComplexDPGWeakFrom::AddDomainLFIntegrator: test fecol index out of bounds");
   if (lfi_r)
   {
      lfis_r[n]->Append(lfi_r);
   }
   if (lfi_i)
   {
      lfis_i[n]->Append(lfi_i);
   }
}

void ComplexDPGWeakForm::BuildProlongation()
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
         P->SetBlock(i, i, const_cast<SparseMatrix*>(P_));
         R->SetBlock(i, i, const_cast<SparseMatrix*>(R_));
      }
   }
}

void ComplexDPGWeakForm::ConformingAssemble()
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
void ComplexDPGWeakForm::Assemble(int skip_zeros)
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
   DenseMatrix B_r, Be_r, G_r, Ge_r, A_r;
   DenseMatrix B_i, Be_i, G_i, Ge_i, A_i;
   Vector vec_e_r, vec_r, b_r;
   Vector vec_e_i, vec_i, b_i;
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
         MFEM_ABORT("ComplexDPGWeakForm::Assemble: dim > 3 not supported");
      }
      int numfaces = faces.Size();

      Array<int> test_offs(test_fecols.Size()+1); test_offs[0] = 0;
      Array<int> trial_offs(trial_fes.Size()+1); trial_offs = 0;

      eltrans = mesh->GetElementTransformation(iel);
      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder(); // assuming uniform order
         test_offs[j+1] = test_fecols_vdims[j]*test_fecols[j]->GetFE(
                             eltrans->GetGeometryType(), order)->GetDof();
      }
      for (int j = 0; j < trial_fes.Size(); j++)
      {
         if (IsTraceFes[j])
         {
            for (int ie = 0; ie < faces.Size(); ie++)
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

      G_r.SetSize(test_offs.Last()); G_r = 0.0;
      vec_r.SetSize(test_offs.Last()); vec_r = 0.0;
      B_r.SetSize(test_offs.Last(),trial_offs.Last()); B_r = 0.0;
      G_i.SetSize(test_offs.Last()); G_i = 0.0;
      vec_i.SetSize(test_offs.Last()); vec_i = 0.0;
      B_i.SetSize(test_offs.Last(),trial_offs.Last()); B_i = 0.0;

      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order_j = test_fecols[j]->GetOrder();

         eltrans = mesh->GetElementTransformation(iel);
         const FiniteElement & test_fe =
            *test_fecols[j]->GetFE(eltrans->GetGeometryType(), order_j);

         // real integrators
         for (int k = 0; k < lfis_r[j]->Size(); k++)
         {
            (*lfis_r[j])[k]->AssembleRHSElementVect(test_fe, *eltrans, vec_e_r);
            vec_r.AddSubVector(vec_e_r, test_offs[j]);
         }
         // imag integrators
         for (int k = 0; k < lfis_i[j]->Size(); k++)
         {
            (*lfis_i[j])[k]->AssembleRHSElementVect(test_fe,*eltrans,vec_e_i);
            vec_i.AddSubVector(vec_e_i, test_offs[j]);
         }

         for (int i = 0; i < test_fecols.Size(); i++)
         {
            int order_i = test_fecols[i]->GetOrder();
            eltrans = mesh->GetElementTransformation(iel);
            const FiniteElement & test_fe_i =
               *test_fecols[i]->GetFE(eltrans->GetGeometryType(), order_i);

            // real integrators
            for (int k = 0; k < test_integs_r(i,j)->Size(); k++)
            {
               if (i==j)
               {
                  (*test_integs_r(i,j))[k]->AssembleElementMatrix(test_fe, *eltrans, Ge_r);
               }
               else
               {
                  (*test_integs_r(i,j))[k]->AssembleElementMatrix2(test_fe_i, test_fe, *eltrans,
                                                                   Ge_r);
               }
               G_r.AddSubMatrix(test_offs[j], test_offs[i], Ge_r);
            }

            // imag integrators
            for (int k = 0; k < test_integs_i(i,j)->Size(); k++)
            {
               if (i==j)
               {
                  (*test_integs_i(i,j))[k]->AssembleElementMatrix(test_fe,*eltrans,Ge_i);
               }
               else
               {
                  (*test_integs_i(i,j))[k]->AssembleElementMatrix2(test_fe_i,test_fe,*eltrans,
                                                                   Ge_i);
               }
               G_i.AddSubMatrix(test_offs[j], test_offs[i], Ge_i);
            }
         }

         for (int i = 0; i < trial_fes.Size(); i++)
         {
            if (IsTraceFes[i])
            {
               // real integrators
               for (int k = 0; k < trial_integs_r(i,j)->Size(); k++)
               {
                  int face_dof_offs = 0;
                  for (int ie = 0; ie < numfaces; ie++)
                  {
                     int iface = faces[ie];
                     FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
                     const FiniteElement & tfe = *trial_fes[i]->GetFaceElement(iface);
                     (*trial_integs_r(i,j))[k]->AssembleTraceFaceMatrix(iel, tfe, test_fe, *ftr,
                                                                        Be_r);
                     B_r.AddSubMatrix(test_offs[j], trial_offs[i]+face_dof_offs, Be_r);
                     face_dof_offs += Be_r.Width();
                  }
               }
               // imag integrators
               for (int k = 0; k < trial_integs_i(i,j)->Size(); k++)
               {
                  int face_dof_offs = 0;
                  for (int ie = 0; ie < numfaces; ie++)
                  {
                     int iface = faces[ie];
                     FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
                     const FiniteElement & tfe = *trial_fes[i]->GetFaceElement(iface);
                     (*trial_integs_i(i,j))[k]->AssembleTraceFaceMatrix(iel,tfe,test_fe,*ftr,Be_i);
                     B_i.AddSubMatrix(test_offs[j], trial_offs[i]+face_dof_offs, Be_i);
                     face_dof_offs += Be_i.Width();
                  }
               }
            }
            else
            {
               const FiniteElement & fe = *trial_fes[i]->GetFE(iel);
               eltrans = mesh->GetElementTransformation(iel);
               // real integrators
               for (int k = 0; k < trial_integs_r(i,j)->Size(); k++)
               {
                  (*trial_integs_r(i,j))[k]->AssembleElementMatrix2(fe,test_fe,*eltrans,Be_r);
                  B_r.AddSubMatrix(test_offs[j], trial_offs[i], Be_r);
               }
               // imag integrators
               for (int k = 0; k < trial_integs_i(i,j)->Size(); k++)
               {
                  (*trial_integs_i(i,j))[k]->AssembleElementMatrix2(fe, test_fe, *eltrans, Be_i);
                  B_i.AddSubMatrix(test_offs[j], trial_offs[i], Be_i);
               }
            }
         }
      }

      ComplexCholeskyFactors chol(G_r.GetData(), G_i.GetData());
      int h = G_r.Height();
      chol.Factor(h);

      int w = B_r.Width();
      chol.LSolve(h,w,B_r.GetData(), B_i.GetData());
      chol.LSolve(h,1,vec_r.GetData(), vec_i.GetData());

      Vector vec(vec_i.Size()+vec_r.Size());
      vec.SetVector(vec_r, 0);
      vec.SetVector(vec_i, vec_r.Size());

      if (store_matrices)
      {
         Bmat[iel] = new ComplexDenseMatrix(new DenseMatrix(B_r), new DenseMatrix(B_i),
                                            true,true);
         fvec[iel] = new Vector(vec);
      }
      ComplexDenseMatrix B(&B_r, &B_i, false, false);
      ComplexDenseMatrix * A = mfem::MultAtB(B, B);
      Vector b(B.Width());
      B.MultTranspose(vec, b);

      b_r.MakeRef(b, 0, b.Size()/2);
      b_i.MakeRef(b, b.Size()/2,b.Size()/2);

      if (static_cond)
      {
         static_cond->AssembleReducedSystem(iel,*A,b_r,b_i);
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
            for (int j = 0; j < trial_fes.Size(); j++)
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

               DenseMatrix Ae_r, Ae_i;
               A->real().GetSubMatrix(trial_offs[i],trial_offs[i+1],
                                      trial_offs[j],trial_offs[j+1], Ae_r);
               A->imag().GetSubMatrix(trial_offs[i],trial_offs[i+1],
                                      trial_offs[j],trial_offs[j+1], Ae_i);
               if (doftrans_i || doftrans_j)
               {
                  TransformDual(doftrans_i, doftrans_j, Ae_r);
                  TransformDual(doftrans_i, doftrans_j, Ae_i);
               }
               if (!mat_r)
               {
                  mfem::out << "null matrix " << std::endl;
               }
               mat_r->GetBlock(i,j).AddSubMatrix(vdofs_i,vdofs_j, Ae_r);
               mat_i->GetBlock(i,j).AddSubMatrix(vdofs_i,vdofs_j, Ae_i);
            }

            // assemble rhs
            Vector vec1_r(b_r,trial_offs[i],trial_offs[i+1]-trial_offs[i]);
            Vector vec1_i(b_i,trial_offs[i],trial_offs[i+1]-trial_offs[i]);

            if (doftrans_i)
            {
               doftrans_i->TransformDual(vec1_r);
               doftrans_i->TransformDual(vec1_i);
            }
            y_r->GetBlock(i).AddElementVector(vdofs_i,vec1_r);
            y_i->GetBlock(i).AddElementVector(vdofs_i,vec1_i);
         }
      }
      delete A;
   } // end of loop through elements
}

void ComplexDPGWeakForm::FormLinearSystem(const Array<int>
                                          &ess_tdof_list,
                                          Vector &x,
                                          OperatorHandle &A,
                                          Vector &X,
                                          Vector &B,
                                          int copy_interior)
{
   FormSystemMatrix(ess_tdof_list, A);
   if (static_cond)
   {
      static_cond->ReduceSystem(x, X, B, copy_interior);
   }
   else if (!P)
   {
      Vector x_r(x, 0, x.Size()/2);
      Vector x_i(x, x.Size()/2, x.Size()/2);
      EliminateVDofsInRHS(ess_tdof_list, x_r,x_i, *y_r, *y_i);
      if (!copy_interior)
      {
         x_r.SetSubVectorComplement(ess_tdof_list, 0.0);
         x_i.SetSubVectorComplement(ess_tdof_list, 0.0);
      }
      X.MakeRef(x, 0, x.Size());
      B.MakeRef(*y,0,y->Size());
   }
   else // non conforming space
   {
      B.SetSize(2*P->Width());
      Vector B_r(B, 0, P->Width());
      Vector B_i(B, P->Width(),P->Width());

      P->MultTranspose(*y_r, B_r);
      P->MultTranspose(*y_i, B_i);
      Vector tmp_r,tmp_i;
      for (int i = 0; i<nblocks; i++)
      {
         if (P->IsZeroBlock(i,i))
         {
            int offset = tdof_offsets[i];
            tmp_r.MakeRef(*y_r, offset,tdof_offsets[i+1]-tdof_offsets[i]);
            tmp_i.MakeRef(*y_i, offset,tdof_offsets[i+1]-tdof_offsets[i]);
            B_r.SetVector(tmp_r,offset);
            B_i.SetVector(tmp_i,offset);
         }
      }

      X.SetSize(2*R->Height());
      Vector X_r(X, 0, X.Size()/2);
      Vector X_i(X, X.Size()/2, X.Size()/2);

      Vector x_r(x, 0,x.Size()/2);
      Vector x_i(x, x.Size()/2, x.Size()/2);

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

void ComplexDPGWeakForm::FormSystemMatrix(const Array<int>
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
      A.Reset(&static_cond->GetSchurComplexOperator(), false);
   }
   else
   {
      if (!mat_e_r)
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
      mat = new ComplexOperator(mat_r,mat_i,false,false);
      A.Reset(mat,false);
   }
}

void ComplexDPGWeakForm::EliminateVDofsInRHS(
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

void ComplexDPGWeakForm::EliminateVDofs(const Array<int> &vdofs,
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

void ComplexDPGWeakForm::RecoverFEMSolution(const Vector &X,
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

void ComplexDPGWeakForm::ReleaseInitMemory()
{
   if (initialized)
   {
      for (int k = 0; k < trial_integs_r.NumRows(); k++)
      {
         for (int l = 0; l < trial_integs_r.NumCols(); l++)
         {
            for (int i = 0; i < trial_integs_r(k,l)->Size(); i++)
            {
               delete (*trial_integs_r(k,l))[i];
            }
            delete trial_integs_r(k,l);
            for (int i = 0; i < trial_integs_i(k,l)->Size(); i++)
            {
               delete (*trial_integs_i(k,l))[i];
            }
            delete trial_integs_i(k,l);
         }
      }
      trial_integs_r.DeleteAll();
      trial_integs_i.DeleteAll();

      for (int k = 0; k < test_integs_r.NumRows(); k++)
      {
         for (int l = 0; l < test_integs_r.NumCols(); l++)
         {
            for (int i = 0; i < test_integs_r(k,l)->Size(); i++)
            {
               delete (*test_integs_r(k,l))[i];
            }
            delete test_integs_r(k,l);
            for (int i = 0; i < test_integs_i(k,l)->Size(); i++)
            {
               delete (*test_integs_i(k,l))[i];
            }
            delete test_integs_i(k,l);
         }
      }
      test_integs_r.DeleteAll();
      test_integs_i.DeleteAll();

      for (int k = 0; k < lfis_r.Size(); k++)
      {
         for (int i = 0; i < lfis_r[k]->Size(); i++)
         {
            delete (*lfis_r[k])[i];
         }
         delete lfis_r[k];
         for (int i = 0; i < lfis_i[k]->Size(); i++)
         {
            delete (*lfis_i[k])[i];
         }
         delete lfis_i[k];
      }
      lfis_r.DeleteAll();
      lfis_i.DeleteAll();
   }
}

void ComplexDPGWeakForm::Update()
{
   delete mat_e_r; mat_e_r = nullptr;
   delete mat_e_i; mat_e_i = nullptr;
   delete mat; mat = nullptr;
   delete mat_r; mat_r = nullptr;
   delete mat_i; mat_i = nullptr;
   delete y; y = nullptr;
   delete y_r; y_r = nullptr;
   delete y_i; y_i = nullptr;

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
      for (int i = 0; i < Bmat.Size(); i++)
      {
         delete Bmat[i]; Bmat[i] = nullptr;
         delete fvec[i]; fvec[i] = nullptr;
      }
      Bmat.SetSize(mesh->GetNE());
      fvec.SetSize(mesh->GetNE());
      for (int i = 0; i < Bmat.Size(); i++)
      {
         Bmat[i] = nullptr;
         fvec[i] = nullptr;
      }
   }
}

void ComplexDPGWeakForm::EnableStaticCondensation()
{
   delete static_cond;
   static_cond = new ComplexBlockStaticCondensation(trial_fes);
}

Vector & ComplexDPGWeakForm::ComputeResidual(const Vector & x)
{
   MFEM_VERIFY(store_matrices,
               "Matrices needed for the residual are not store. Call ComplexDPGWeakForm::StoreMatrices()")
   // wrap vector in a blockvector
   int n = x.Size()/2;

   BlockVector x_r(const_cast<Vector &>(x),0,dof_offsets);
   BlockVector x_i(const_cast<Vector &>(x),n,dof_offsets);

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
         MFEM_ABORT("ComplexDPGWeakForm::ComputeResidual: "
                    "dim > 3 not supported");
      }
      int numfaces = faces.Size();

      Array<int> trial_offs(trial_fes.Size()+1); trial_offs = 0;

      for (int j = 0; j < trial_fes.Size(); j++)
      {
         if (IsTraceFes[j])
         {
            for (int ie = 0; ie < faces.Size(); ie++)
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

      int nn = trial_offs.Last();
      u.SetSize(2*nn);
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
         Vector vec1_r;
         Vector vec1_i;
         vec1_r.MakeRef(u, trial_offs[i], trial_offs[i+1]-trial_offs[i]);
         vec1_i.MakeRef(u, trial_offs[i]+nn, trial_offs[i+1]-trial_offs[i]);
         x_r.GetBlock(i).GetSubVector(vdofs,vec1_r);
         x_i.GetBlock(i).GetSubVector(vdofs,vec1_i);
         if (doftrans)
         {
            doftrans->InvTransformPrimal(vec1_r);
            doftrans->InvTransformPrimal(vec1_i);
         }
      } // end of loop through trial spaces

      // residual
      Vector v(Bmat[iel]->Height());
      Bmat[iel]->Mult(u,v);
      v -= *fvec[iel];
      residuals[iel] = v.Norml2();
   } // end of loop through elements
   return residuals;
}

ComplexDPGWeakForm::~ComplexDPGWeakForm()
{
   delete mat_e_r; mat_e_r = nullptr;
   delete mat_e_i; mat_e_i = nullptr;
   delete mat; mat = nullptr;
   delete mat_r; mat_r = nullptr;
   delete mat_i; mat_i = nullptr;
   delete y; y = nullptr;
   delete y_r; y_r = nullptr;
   delete y_i; y_i = nullptr;

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
