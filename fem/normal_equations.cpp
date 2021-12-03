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

void NormalEquations::Init()
{
   domain_integs.SetSize(domain_fes.Size(), test_fecols.Size());
   for (int i = 0; i < domain_integs.NumRows(); i++)
   {
      for (int j = 0; j < domain_integs.NumCols(); j++)
      {
         domain_integs(i,j) = new Array<BilinearFormIntegrator * >();
      }
   }

   trace_integs.SetSize(trace_fes.Size(), test_fecols.Size());
   for (int i = 0; i < trace_integs.NumRows(); i++)
   {
      for (int j = 0; j < trace_integs.NumCols(); j++)
      {
         trace_integs(i,j) = new Array<BilinearFormIntegrator * >();
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

   domain_lf_integs.SetSize(test_fecols.Size());
   for (int j = 0; j < domain_lf_integs.Size(); j++)
   {
      domain_lf_integs[j] = new Array<LinearFormIntegrator * >();
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
   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;
}

// Allocate SparseMatrix and RHS
void NormalEquations::AllocMat()
{
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

void NormalEquations::Finalize(int skip_zeros)
{
   if (mat)
   {
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            mat->GetBlock(i,j).Finalize(skip_zeros);
         }
      }
   }

   if (mat_e)
   {
      for (int i = 0; i<nblocks; i++)
      {
         for (int j = 0; j<nblocks; j++)
         {
            mat_e->GetBlock(i,j).Finalize(skip_zeros);
         }
      }
   }
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void NormalEquations::AddDomainBFIntegrator(
   BilinearFormIntegrator *bfi, int trial_fes, int test_fes)
{
   domain_integs(trial_fes,test_fes)->Append(bfi);
}

/// Adds new Domain BF Integrator. Assumes ownership of @a bfi.
void NormalEquations::AddTestIntegrator
(BilinearFormIntegrator *bfi, int test_fes0, int test_fes1)
{
   test_integs(test_fes0,test_fes1)->Append(bfi);
}

/// Adds new Trace element BF Integrator. Assumes ownership of @a bfi.
void NormalEquations::AddTraceElementBFIntegrator(
   BilinearFormIntegrator * bfi, int trial_fes, int test_fes)
{
   trace_integs(trial_fes,test_fes)->Append(bfi);
}

/// Adds new Domain LF Integrator. Assumes ownership of @a bfi.
void NormalEquations::AddDomainLFIntegrator(
   LinearFormIntegrator *lfi, int test_fes)
{
   domain_lf_integs[test_fes]->Append(lfi);
}

void NormalEquations::BuildProlongation()
{
   P = new BlockMatrix(dof_offsets, tdof_offsets);
   R = new BlockMatrix(tdof_offsets, dof_offsets);
   for (int i = 0; i<nblocks; i++)
   {
      const SparseMatrix *P_ = fespaces[i]->GetConformingProlongation();
      if (P_)
      {
         const SparseMatrix *R_ = fespaces[i]->GetRestrictionMatrix();
         P->SetBlock(i,i,const_cast<SparseMatrix*>(P_));
         R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
      }
      else
      {
         // TODO improve this by using BlockOperator/BlockMatrix
         Vector diag(fespaces[i]->GetVSize()); diag = 1.;
         P->SetBlock(i,i,new SparseMatrix(diag));
         R->SetBlock(i,i,new SparseMatrix(diag));
      }
   }
}

void NormalEquations::ConformingAssemble()
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA = mfem::Mult(*Pt, *mat);
   delete mat;
   if (mat_e)
   {
      BlockMatrix *PtAe = mfem::Mult(*Pt, *mat_e);
      delete mat_e;
      mat_e = PtAe;
   }
   delete Pt;
   mat = mfem::Mult(*PtA, *P);

   if (mat_e)
   {
      BlockMatrix *PtAeP = mfem::Mult(*mat_e, *P);
      delete mat_e;
      mat_e = PtAeP;
   }
   height = mat->Height();
   width = mat->Width();
}

/// Assembles the form i.e. sums over all domain integrators.
void NormalEquations::Assemble(int skip_zeros)
{
   ElementTransformation *eltrans;
   Array<int> faces, ori;

   // DofTransformation * doftrans_j, *doftrans_k;
   Array<int> vdofs_j;
   Array<int> offsetvdofs_j;
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
      if (dim == 2)
      {
         mesh->GetElementEdges(iel, faces, ori);
      }
      else //dim = 3
      {
         mesh->GetElementFaces(iel,faces,ori);
      }
      int numfaces = faces.Size();

      Array<int> test_offs(test_fecols.Size()+1); test_offs[0] = 0;
      Array<int> domain_offs(domain_fes.Size()+1); domain_offs[0] = 0;
      Array<int> trace_offs(trace_fes.Size()+1); trace_offs = 0;

      eltrans = mesh->GetElementTransformation(iel);
      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder(); // assuming uniform order
         test_offs[j+1] = test_fecols[j]->GetFE(eltrans->GetGeometryType(),
                                                order)->GetDof();
      }
      for (int j = 0; j < domain_fes.Size(); j++)
      {
         domain_offs[j+1] = domain_fes[j]->GetVDim() * domain_fes[j]->GetFE(
                               iel)->GetDof();
      }
      for (int j = 0; j < trace_fes.Size(); j++)
      {
         for (int ie = 0; ie<faces.Size(); ie++)
         {
            trace_offs[j+1] += trace_fes[j]->GetFaceElement(faces[ie])->GetDof();
         }
      }
      test_offs.PartialSum();
      domain_offs.PartialSum();
      trace_offs.PartialSum();

      G.SetSize(test_offs.Last()); G = 0.0;
      vec.SetSize(test_offs.Last()); vec = 0.0;
      B.SetSize(test_offs.Last(),domain_offs.Last()+trace_offs.Last()); B = 0.0;


      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder();

         eltrans = mesh->GetElementTransformation(iel);
         const FiniteElement & test_fe =
            *test_fecols[j]->GetFE(eltrans->GetGeometryType(), order);

         for (int k = 0; k < domain_lf_integs[j]->Size(); k++)
         {
            (*domain_lf_integs[j])[k]->AssembleRHSElementVect(test_fe,
                                                              *eltrans,vec_e);
            vec.AddSubVector(vec_e,test_offs[j]);
         }

         for (int i = 0; i < test_fecols.Size(); i++)
         {
            int order = test_fecols[i]->GetOrder();
            eltrans = mesh->GetElementTransformation(iel);
            const FiniteElement & test_fe_i =
               *test_fecols[i]->GetFE(eltrans->GetGeometryType(), order);

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

         for (int i = 0; i < domain_fes.Size(); i++)
         {
            const FiniteElement & fe = *domain_fes[i]->GetFE(iel);
            eltrans = mesh->GetElementTransformation(iel);
            for (int k = 0; k < domain_integs(i,j)->Size(); k++)
            {
               (*domain_integs(i,j))[k]->AssembleElementMatrix2(fe,test_fe,*eltrans,Be);
               B.AddSubMatrix(test_offs[j], domain_offs[i], Be);
            }
         }
         for (int i = 0; i < trace_fes.Size(); i++)
         {
            for (int k = 0; k < trace_integs(i,j)->Size(); k++)
            {
               int face_dof_offs = 0;
               for (int ie = 0; ie < numfaces; ie++)
               {
                  int iface = faces[ie];
                  FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
                  const FiniteElement & tfe = *trace_fes[i]->GetFaceElement(iface);
                  (*trace_integs(i,j))[k]->AssembleTraceFaceMatrix(iel,tfe,test_fe,*ftr,Be);
                  B.AddSubMatrix(test_offs[j], domain_offs.Last()+trace_offs[i]+face_dof_offs,
                                 Be);

                  face_dof_offs+=Be.Width();
               }
            }
         }
      }

      G.Invert();

      // Form Normal Equations B^T G^-1 B = B^T G^-1 l
      RAP(G,B,A);

      Gvec.SetSize(G.Height());
      G.Mult(vec,Gvec);
      b.SetSize(B.Width());
      B.MultTranspose(Gvec,b);

      // Assembly
      for (int i = 0; i<fespaces.Size(); i++)
      {
         int ibeg, iend;
         Array<int> vdofs_i;
         DofTransformation * doftrans_i = nullptr;
         DofTransformation * doftrans_j = nullptr;
         if (i<domain_fes.Size())
         {
            doftrans_i = domain_fes[i]->GetElementVDofs(iel, vdofs_i);
            ibeg = domain_offs[i];
            iend = domain_offs[i+1];
         }
         else
         {
            Array<int> face_vdofs;
            for (int k = 0; k < numfaces; k++)
            {
               int iface = faces[k];
               trace_fes[i-domain_fes.Size()]->GetFaceVDofs(iface, face_vdofs);
               vdofs_i.Append(face_vdofs);
            }
            ibeg = domain_offs.Last() + trace_offs[i-domain_fes.Size()];
            iend = domain_offs.Last() + trace_offs[i+1-domain_fes.Size()];
         }
         for (int j = 0; j<fespaces.Size(); j++)
         {
            Array<int> vdofs_j;
            int jbeg, jend;
            if (j<domain_fes.Size())
            {
               doftrans_j = domain_fes[j]->GetElementVDofs(iel, vdofs_j);
               jbeg = domain_offs[j];
               jend = domain_offs[j+1];
            }
            else
            {
               Array<int> face_vdofs;
               for (int k = 0; k < numfaces; k++)
               {
                  int iface = faces[k];
                  trace_fes[j-domain_fes.Size()]->GetFaceVDofs(iface, face_vdofs);
                  vdofs_j.Append(face_vdofs);
               }
               jbeg = domain_offs.Last() + trace_offs[j-domain_fes.Size()];
               jend = domain_offs.Last() + trace_offs[j+1-domain_fes.Size()];
            }

            DenseMatrix Ae;
            A.GetSubMatrix(ibeg,iend, jbeg,jend, Ae);
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
         vec1.SetDataAndSize(&data[ibeg], iend-ibeg);
         if (doftrans_i)
         {
            doftrans_i->TransformDual(vec1);
         }
         y->GetBlock(i).AddElementVector(vdofs_i,vec1);
      }
   }
}

void NormalEquations::FormLinearSystem(const Array<int>
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

void NormalEquations::FormSystemMatrix(const Array<int>
                                       &ess_tdof_list,
                                       OperatorHandle &A)
{
   if (!mat_e)
   {
      bool conforming = true;
      for (int i = 0; i<nblocks; i++)
      {
         const SparseMatrix *P_ = fespaces[i]->GetConformingProlongation();
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

void NormalEquations::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x,b,-1.);
   Array<int> cols;
   Vector srow;
   for (int i = 0; i<vdofs.Size(); i++)
   {
      int dof = (vdofs[i]>=0) ? vdofs[i] : -1-vdofs[i];
      mat->GetRow(dof,cols,srow);

      double s=0.0;
      for (int k = 0; k <cols.Size(); k++)
      {
         s += srow[k] * x[cols[k]];
      }
      b[dof] = s;
   }
}

void NormalEquations::EliminateVDofs(const Array<int> &vdofs,
                                     Operator::DiagonalPolicy dpolicy)
{
   // Alternative elimination of essential dofs using the BlockMatrix
   if (mat_e == NULL)
   {
      Array<int> offsets;

      if (P)
      {
         offsets.MakeRef(tdof_offsets);
      }
      else
      {
         offsets.MakeRef(tdof_offsets);
      }
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

void NormalEquations::RecoverFEMSolution(const Vector &X,
                                         Vector &x)
{
   if (!P)
   {
      x.SyncMemory(X);
   }
   else
   {
      x.SetSize(P->Height());
      P->Mult(X, x);
   }
}

NormalEquations::~NormalEquations()
{
   delete mat_e;
   delete mat;
   delete y;

   for (int k = 0; k< domain_integs.NumRows(); k++)
   {
      for (int l = 0; l<domain_integs.NumCols(); l++)
      {
         for (int i = 0; i<domain_integs(k,l)->Size(); i++)
         {
            delete (*domain_integs(k,l))[i];
         }
         delete domain_integs(k,l);
      }
   }

   for (int k = 0; k< trace_integs.NumRows(); k++)
   {
      for (int l = 0; l<trace_integs.NumCols(); l++)
      {
         for (int i = 0; i<trace_integs(k,l)->Size(); i++)
         {
            delete (*trace_integs(k,l))[i];
         }
         delete trace_integs(k,l);
      }
   }

   for (int k = 0; k< test_integs.NumRows(); k++)
   {
      for (int l = 0; l<test_integs.NumCols(); l++)
      {
         for (int i = 0; i<test_integs(k,l)->Size(); i++)
         {
            delete (*test_integs(k,l))[i];
         }
         delete test_integs(k,l);
      }
   }

   if (P)
   {
      for (int i = 0; i<nblocks; i++)
      {
         const SparseMatrix *P_ = fespaces[i]->GetConformingProlongation();
         if (!P_)
         {
            delete &P->GetBlock(i,i);
            delete &R->GetBlock(i,i);
         }
      }
      delete P;
      delete R;
   }

   if (store_mat)
   {
      for (int i = 0; i<mesh->GetNE(); i++)
      {
         delete  GB[i];
         delete  Gl[i];
      }
   }
}

} // namespace mfem
