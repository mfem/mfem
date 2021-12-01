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
   mat = new SparseMatrix(height);
   y = new Vector(height);
   *y = 0.;
}

void NormalEquations::Finalize(int skip_zeros)
{
   mat->Finalize(skip_zeros);
   if (mat_e) { mat_e->Finalize(skip_zeros); }
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
                  B.AddSubMatrix(test_offs[j], domain_offs.Last()+trace_offs[i]+face_dof_offs, Be);
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
      vdofs.SetSize(0);
      for (int j = 0; j<domain_fes.Size(); j++)
      {
         DofTransformation * doftrans = domain_fes[j]->GetElementVDofs(iel, vdofs_j);
         if (doftrans)
         {
            MFEM_ABORT("NormalEquations::Assemble(): doftrans not implemented yet")
         }
         int offset_j = dof_offsets[j];
         offsetvdofs_j.SetSize(vdofs_j.Size());
         for (int l = 0; l<vdofs_j.Size(); l++)
         {
            offsetvdofs_j[l] = vdofs_j[l]<0 ? -offset_j + vdofs_j[l]
                               :  offset_j + vdofs_j[l];
         }
         vdofs.Append(offsetvdofs_j);
      }

      for (int j = domain_fes.Size(); j<fespaces.Size(); j++)
      {
         int offset_j = dof_offsets[j];
         Array<int> face_vdofs;
         for (int k = 0; k < numfaces; k++)
         {
            int iface = faces[k];
            fespaces[j]->GetFaceVDofs(iface, vdofs_j);
            face_vdofs.Append(vdofs_j);
         }
         offsetvdofs_j.SetSize(face_vdofs.Size());
         for (int l = 0; l<face_vdofs.Size(); l++)
         {
            offsetvdofs_j[l] = face_vdofs[l]<0 ? -offset_j + face_vdofs[l]
                               :  offset_j + face_vdofs[l];
         }
         vdofs.Append(offsetvdofs_j);
      }

      mat->AddSubMatrix(vdofs,vdofs,A, skip_zeros);
      y->AddElementVector(vdofs,b);
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
      EliminateVDofs(ess_tdof_list, diag_policy);
      const int remove_zeros = 0;
      Finalize(remove_zeros);
   }
   A.Reset(mat, false);
}

void NormalEquations::EliminateVDofsInRHS(
   const Array<int> &vdofs, const Vector &x, Vector &b)
{
   mat_e->AddMult(x, b, -1.);
   mat->PartMult(vdofs, x, b);
}

void NormalEquations::EliminateVDofs(const Array<int> &vdofs,
                                     Operator::DiagonalPolicy dpolicy)
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
