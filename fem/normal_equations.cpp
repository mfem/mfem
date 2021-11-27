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
   // Initialize
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
   element_matrices = NULL;
   diag_policy = mfem::Operator::DIAG_ONE;
   height = dof_offsets[nblocks];
   width = height;
}


// Allocate appropriate SparseMatrix and assign it to mat
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

/// Adds new Trance element BF Integrator. Assumes ownership of @a bfi.
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
      const SparseMatrix *R_ = fespaces[i]->GetRestrictionMatrix();
      P->SetBlock(i,i,const_cast<SparseMatrix*>(P_));
      R->SetBlock(i,i,const_cast<SparseMatrix*>(R_));
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
   DenseMatrix Bf, G, ElemG,ElemBf, ElemBh;
   DenseMatrix BlkB, BlkG;
   Vector elvec, elvect, blockvec;

   // loop through elements
   for (int iel = 0; iel < mesh -> GetNE(); iel++)
   {

      if (dim == 2)
      {
         mesh->GetElementEdges(iel, faces, ori);
      }
      else if (dim == 3)
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
         // assuming uniform order
         int order = test_fecols[j]->GetOrder();
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

      BlkG.SetSize(test_offs.Last()); BlkG = 0.0;
      blockvec.SetSize(test_offs.Last()); blockvec = 0.0;
      BlkB.SetSize(test_offs.Last(),domain_offs.Last()+trace_offs.Last()); BlkB = 0.0;


      // loop through test fe spaces
      for (int j = 0; j < test_fecols.Size(); j++)
      {
         int order = test_fecols[j]->GetOrder();

         eltrans = mesh->GetElementTransformation(iel);
         const FiniteElement & test_fe =
            *test_fecols[j]->GetFE(eltrans->GetGeometryType(), order);

         // RHS vector
         elvec.SetSize(0);
         for (int k = 0; k < domain_lf_integs[j]->Size(); k++)
         {
            (*domain_lf_integs[j])[k]->AssembleRHSElementVect(test_fe,
                                                              *eltrans,elvect);
            if (elvec.Size() == 0)
            {
               elvec = elvect;
            }
            else
            {
               elvec += elvect;
            }
         }
         // set the block vector
         blockvec.SetVector(elvec,test_offs[j]);

         // Test space integrators
         for (int i = 0; i < test_fecols.Size(); i++)
         {
            int order = test_fecols[i]->GetOrder();
            eltrans = mesh->GetElementTransformation(iel);
            const FiniteElement & test_fe_i =
               *test_fecols[i]->GetFE(eltrans->GetGeometryType(), order);

            // loop though test_integ for this combination
            G.SetSize(0);
            for (int k = 0; k < test_integs(i,j)->Size(); k++)
            {
               if (i==j)
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix(test_fe,*eltrans,ElemG);
               }
               else
               {
                  (*test_integs(i,j))[k]->AssembleElementMatrix2(test_fe_i,test_fe,*eltrans,
                                                                 ElemG);
               }
               if (G.Size() == 0)
               {
                  G = ElemG;
               }
               else
               {
                  G += ElemG;
               }
            }
            BlkG.SetSubMatrix(test_offs[j], test_offs[i], G);
         } // end of 2nd loop though test spaces

         // Field (domain) integrators
         for (int i = 0; i < domain_fes.Size(); i++)
         {
            const FiniteElement & fe = *domain_fes[i]->GetFE(iel);
            // loop though domain_integs
            Bf.SetSize(0);
            eltrans = mesh->GetElementTransformation(iel);
            for (int k = 0; k < domain_integs(i,j)->Size(); k++)
            {
               (*domain_integs(i,j))[k]->AssembleElementMatrix2(fe,test_fe,*eltrans,ElemBf);
               if (Bf.Size() == 0)
               {
                  Bf = ElemBf;
               }
               else
               {
                  Bf += ElemBf;
               }
            }
            BlkB.SetSubMatrix(test_offs[j], domain_offs[i], Bf);
         } // end of loop through domain trial spaces


         // Trace integrators
         for (int i = 0; i < trace_fes.Size(); i++)
         {
            ElemBh.SetSize(test_offs[j+1] - test_offs[j], trace_offs[i+1] - trace_offs[i]);
            Array<DenseMatrix * > Baux(numfaces);
            for (int ie = 0; ie < numfaces; ie++)
            {
               Baux[ie] = new DenseMatrix(0);
            }
            for (int k = 0; k < trace_integs(i,j)->Size(); k++)
            {
               for (int ie = 0; ie < numfaces; ie++)
               {
                  int iface = faces[ie];
                  FaceElementTransformations * ftr = mesh->GetFaceElementTransformations(iface);
                  const FiniteElement & tfe = *trace_fes[i]->GetFaceElement(iface);
                  DenseMatrix aux;
                  (*trace_integs(i,j))[k]->AssembleTraceFaceMatrix(iel,tfe,test_fe,*ftr,aux);
                  if (Baux[ie]->Size() == 0)
                  {
                     *Baux[ie] = aux;
                  }
                  else
                  {
                     *Baux[ie] += aux;
                  }
               }
            }
            ElemBh.SetSubMatrix(0,0,*Baux[0]);
            int jbeg = Baux[0]->Width();
            for (int ie = 1; ie < numfaces; ie++)
            {
               ElemBh.SetSubMatrix(0,jbeg,*Baux[ie]);
               jbeg += Baux[ie]->Width();
            }
            for (int ie = 0; ie < numfaces; ie++)
            {
               delete Baux[ie];
            }
            BlkB.SetSubMatrix(test_offs[j], domain_offs.Last()+trace_offs[i], ElemBh);
         }  // end of loop through trace trial spaces
      } // end of loop through test spaces


      // Form Normal Equations B^T G^-1 B = B^T G^-1 l
      DenseMatrix A;
      BlkG.Invert();

      RAP(BlkG,BlkB,A);
      Vector b(A.Height());
      Vector Gl(BlkG.Height());
      BlkG.Mult(blockvec,Gl);
      BlkB.MultTranspose(Gl,b);

      vdofs.SetSize(0);
      // field dofs;
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

      // trace dofs;
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
   } // end of loop through elements
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
      const SparseMatrix *P_ = fespaces[0]->GetConformingProlongation();
      if (P_) { ConformingAssemble(); }
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


void NormalEquations::RecoverFEMSolution(const Vector &X,
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

NormalEquations::~NormalEquations()
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
