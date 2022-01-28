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

#include "blockstaticcond.hpp"

namespace mfem
{


BlockStaticCondensation::BlockStaticCondensation(Array<FiniteElementSpace *> &
                                                 fes_)
{
   SetSpaces(fes_);
}

void BlockStaticCondensation::SetSpaces(Array<FiniteElementSpace*> & fes_)
{
   fes = fes_;
   nblocks = fes.Size();
   rblocks = 0;
   tr_fes.SetSize(nblocks);
   IsTraceSpace.SetSize(nblocks);
   mesh = fes[0]->GetMesh();
   const FiniteElementCollection * fec;
   for (int i = 0; i < nblocks; i++)
   {
      fec = fes[i]->FEColl();
      IsTraceSpace[i] =
         (dynamic_cast<const H1_Trace_FECollection*>(fec) ||
          dynamic_cast<const ND_Trace_FECollection*>(fec) ||
          dynamic_cast<const RT_Trace_FECollection*>(fec));

      // skip if it's an L2 space (no trace space to construct)
      tr_fes[i] = (fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) ?
                  nullptr : (IsTraceSpace[i]) ? fes[i] :
                  new FiniteElementSpace(mesh, fec->GetTraceCollection(), fes[i]->GetVDim(),
                                         fes[i]->GetOrdering());
      if (tr_fes[i]) { rblocks++; }
   }

   Init();
}

void BlockStaticCondensation::ComputeOffsets()
{
   rdof_offsets.SetSize(rblocks+1);
   rtdof_offsets.SetSize(rblocks+1);
   rdof_offsets[0] = 0;
   rtdof_offsets[0] = 0;

   int j=0;
   for (int i =0; i<nblocks; i++)
   {
      if (tr_fes[i])
      {
         rdof_offsets[j+1] = tr_fes[i]->GetVSize();
         rtdof_offsets[j+1] = tr_fes[i]->GetTrueVSize();
         j++;
      }
   }
   rdof_offsets.PartialSum();
   rtdof_offsets.PartialSum();
}


void BlockStaticCondensation::Init()
{
   lmat.SetSize(mesh->GetNE());
   lvec.SetSize(mesh->GetNE());


   ComputeOffsets();

   S = new BlockMatrix(rdof_offsets);
   S->owns_blocks = 1;

   for (int i = 0; i<S->NumRowBlocks(); i++)
   {
      int h = rdof_offsets[i+1] - rdof_offsets[i];
      for (int j = 0; j<S->NumColBlocks(); j++)
      {
         int w = rdof_offsets[j+1] - rdof_offsets[j];
         S->SetBlock(i,j,new SparseMatrix(h, w));
      }
   }
   y = new BlockVector(rdof_offsets);
   *y = 0.;
}

void BlockStaticCondensation::GetReduceElementIndicesAndOffsets(int el,
                                                                Array<int> & trace_ldofs,
                                                                Array<int> & interior_ldofs,
                                                                Array<int> & offsets)
{
   int dim = mesh->Dimension();
   offsets.SetSize(tr_fes.Size()+1); offsets = 0;
   Array<int> dofs;
   Array<int> faces, ori;
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else //dim = 3
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   int numfaces = faces.Size();

   trace_ldofs.SetSize(0);
   interior_ldofs.SetSize(0);
   // construct Array of bubble dofs to be extracted
   int skip=0;
   Array<int> tr_dofs;
   Array<int> int_dofs;
   for (int i = 0; i<tr_fes.Size(); i++)
   {
      int td = 0;
      int ndof;
      // if it's an L2 space (bubbles)
      if (!tr_fes[i])
      {
         ndof = fes[i]->GetVDim()*fes[i]->GetFE(el)->GetDof();
         td = 0;
      }
      else if (IsTraceSpace[i])
      {
         for (int iface = 0; iface < numfaces; iface++)
         {
            td += fes[i]->GetVDim()*fes[i]->GetFaceElement(faces[iface])->GetDof();
         }
         ndof = td;
      }
      else
      {
         Array<int> trace_dofs;
         ndof = fes[i]->GetVDim()*fes[i]->GetFE(el)->GetDof();
         tr_fes[i]->GetElementVDofs(el, trace_dofs);
         td = trace_dofs.Size(); // number of trace dofs
      }
      offsets[i+1] = td;
      tr_dofs.SetSize(td);
      int_dofs.SetSize(ndof - td);
      for (int j = 0; j<td; j++)
      {
         tr_dofs[j] = skip + j;
      }
      for (int j = 0; j<ndof-td; j++)
      {
         int_dofs[j] = skip + td + j;
      }
      skip+=ndof;

      trace_ldofs.Append(tr_dofs);
      interior_ldofs.Append(int_dofs);
   }
   offsets.PartialSum();
}


void BlockStaticCondensation::GetLocalShurComplement(int el,
                                                     const Array<int> & tr_idx, const Array<int> & int_idx,
                                                     const DenseMatrix & elmat, const Vector & elvect,
                                                     DenseMatrix & rmat, Vector & rvect)
{
   int rdofs = tr_idx.Size();
   int idofs = int_idx.Size();
   MFEM_VERIFY(idofs != 0, "Number of interior dofs is zero");
   MFEM_VERIFY(rdofs != 0, "Number of interface dofs is zero");

   rmat.SetSize(rdofs);
   rvect.SetSize(rdofs);

   DenseMatrix A_tt, A_ti, A_it, A_ii;
   Vector y_t, y_i;

   elmat.GetSubMatrix(tr_idx,A_tt);
   elmat.GetSubMatrix(tr_idx,int_idx, A_ti);
   elmat.GetSubMatrix(int_idx, tr_idx, A_it);
   elmat.GetSubMatrix(int_idx, A_ii);

   elvect.GetSubVector(tr_idx, y_t);
   elvect.GetSubVector(int_idx, y_i);

   DenseMatrixInverse lu(A_ii);
   lu.Factor();
   lmat[el] = new DenseMatrix(idofs,rdofs);
   lvec[el] = new Vector(idofs);
   DenseMatrix temp_it(idofs,rdofs);

   lu.Mult(A_it,*lmat[el]);
   lu.Mult(y_i,*lvec[el]);

   // LHS
   mfem::Mult(A_ti,*lmat[el],rmat);
   rmat.Neg();
   rmat.Add(1., A_tt);

   // RHS
   A_ti.Mult(*lvec[el], rvect);
   rvect.Neg();
   rvect.Add(1., y_i);
}


void BlockStaticCondensation::AssembleReducedSystem(int el,
                                                    DenseMatrix &elmat,
                                                    Vector & elvect)
{
   // Get Shur Complement
   Array<int> tr_idx, int_idx;
   Array<int> offsets;
   // Get local element idx and offsets for global assembly
   GetReduceElementIndicesAndOffsets(el, tr_idx,int_idx, offsets);

   DenseMatrix rmat, *rmatptr;
   Vector rvec, *rvecptr;
   // Extract the reduced matrices based on tr_idx and int_idx
   if (int_idx.Size()!=0)
   {
      GetLocalShurComplement(el,tr_idx,tr_idx, elmat, elvect, rmat, rvec);
      rmatptr = &rmat;
      rvecptr = &rvec;
   }
   else
   {
      rmatptr = &elmat;
      rvecptr = &elvect;
   }

   // Assemble global mat and rhs
   DofTransformation * doftrans_i, *doftrans_j;

   Array<int> faces, ori;
   int dim = mesh->Dimension();
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else //dim = 3
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   int numfaces = faces.Size();

   int skip_i=0;
   for (int i = 0; i<tr_fes.Size(); i++)
   {
      if (!tr_fes[i]) { continue; }
      Array<int> vdofs_i;
      doftrans_i = nullptr;
      if (IsTraceSpace[i])
      {
         Array<int> face_vdofs;
         for (int k = 0; k < numfaces; k++)
         {
            int iface = faces[k];
            tr_fes[i]->GetFaceVDofs(iface, face_vdofs);
            vdofs_i.Append(face_vdofs);
         }
      }
      else
      {
         doftrans_i = tr_fes[i]->GetElementVDofs(el, vdofs_i);
      }
      int skip_j=0;
      for (int j = 0; j<tr_fes.Size(); j++)
      {
         if (!tr_fes[j]) { continue; }
         Array<int> vdofs_j;
         doftrans_j = nullptr;

         if (IsTraceSpace[j])
         {
            Array<int> face_vdofs;
            for (int k = 0; k < numfaces; k++)
            {
               int iface = faces[k];
               tr_fes[j]->GetFaceVDofs(iface, face_vdofs);
               vdofs_j.Append(face_vdofs);
            }
         }
         else
         {
            doftrans_j = tr_fes[j]->GetElementVDofs(el, vdofs_j);
         }

         DenseMatrix Ae;
         rmatptr->GetSubMatrix(offsets[i],offsets[i+1],
                               offsets[j],offsets[j+1], Ae);
         if (doftrans_i || doftrans_j)
         {
            TransformDual(doftrans_i, doftrans_j, Ae);
         }
         S->GetBlock(skip_i,skip_j).AddSubMatrix(vdofs_i,vdofs_j, Ae);
         skip_j++;
      }

      // assemble rhs
      double * data = rvecptr->GetData();
      Vector vec1;
      // ref subvector
      vec1.SetDataAndSize(&data[offsets[i]],
                          offsets[i+1]-offsets[i]);
      if (doftrans_i)
      {
         doftrans_i->TransformDual(vec1);
      }
      y->GetBlock(skip_i).AddElementVector(vdofs_i,vec1);
      skip_i++;
   }
}

void BlockStaticCondensation::BuildProlongation()
{
   P = new BlockMatrix(rdof_offsets, rtdof_offsets);
   R = new BlockMatrix(rtdof_offsets, rdof_offsets);
   P->owns_blocks = 0;
   R->owns_blocks = 0;
   int skip = 0;
   for (int i = 0; i<nblocks; i++)
   {
      if (!tr_fes[i]) { continue; }
      const SparseMatrix *P_ = tr_fes[i]->GetConformingProlongation();
      if (P_)
      {
         const SparseMatrix *R_ = tr_fes[i]->GetRestrictionMatrix();
         P->SetBlock(skip,skip,const_cast<SparseMatrix*>(P_));
         R->SetBlock(skip,skip,const_cast<SparseMatrix*>(R_));
      }
      skip++;
   }
}


void BlockStaticCondensation::ConformingAssemble(int skip_zeros)
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA = mfem::Mult(*Pt, *S);
   S->owns_blocks = 0;
   for (int i = 0; i<rblocks; i++)
   {
      for (int j = 0; j<rblocks; j++)
      {
         SparseMatrix * tmp = &S->GetBlock(i,j);
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
   delete S;
   if (S_e)
   {
      BlockMatrix *PtAe = mfem::Mult(*Pt, *S_e);
      S_e->owns_blocks = 0;
      for (int i = 0; i<rblocks; i++)
      {
         for (int j = 0; j<rblocks; j++)
         {
            SparseMatrix * tmp = &S_e->GetBlock(i,j);
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
      delete S_e;
      S_e = PtAe;
   }
   delete Pt;

   S = mfem::Mult(*PtA, *P);

   PtA->owns_blocks = 0;
   for (int i = 0; i<rblocks; i++)
   {
      for (int j = 0; j<rblocks; j++)
      {
         SparseMatrix * tmp = &PtA->GetBlock(j,i);
         if (P->IsZeroBlock(i,i))
         {
            S->SetBlock(j,i,tmp);
         }
         else
         {
            delete tmp;
         }
      }
   }
   delete PtA;

   if (S_e)
   {
      BlockMatrix *PtAeP = mfem::Mult(*S_e, *P);
      S_e->owns_blocks = 0;
      for (int i = 0; i<rblocks; i++)
      {
         for (int j = 0; j<rblocks; j++)
         {
            SparseMatrix * tmp = &S_e->GetBlock(j,i);
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

      delete S_e;
      S_e = PtAeP;
   }
   height = S->Height();
   width = S->Width();
}

void BlockStaticCondensation::Finalize(int skip_zeros)
{
   if (S) { S->Finalize(skip_zeros); }
   if (S_e) { S_e->Finalize(skip_zeros); }
}

void BlockStaticCondensation::FormSystemMatrix(Operator::DiagonalPolicy
                                               diag_policy)
{

}




void BlockStaticCondensation::SetEssentialTrueDofs(const Array<int>
                                                   &ess_tdof_list)
{
   // TODO
   // convert list to reduced true dofs
   // 1. convert to global marker for each space
   // 2. booean mult with R for each space
   //



}

void BlockStaticCondensation::EliminateReducedTrueDofs(const Array<int>
                                                       &ess_rtdof_list,
                                                       Matrix::DiagonalPolicy dpolicy)
{

}

void BlockStaticCondensation::EliminateReducedTrueDofs(Matrix::DiagonalPolicy
                                                       dpolicy)
{

}

void BlockStaticCondensation::ReduceRHS(const Vector &b, Vector &sc_b) const
{

}

void BlockStaticCondensation::ReduceSolution(const Vector &sol,
                                             Vector &sc_sol) const
{

}

void BlockStaticCondensation::ReduceSystem(Vector &x, Vector &b, Vector &X,
                                           Vector &B,
                                           int copy_interior) const
{




}

void BlockStaticCondensation::ConvertMarkerToReducedTrueDofs(
   const Array<int> &ess_tdof_marker,
   Array<int> &ess_rtdof_marker) const
{

}

void BlockStaticCondensation::ConvertListToReducedTrueDofs(
   const Array<int> &ess_tdof_list,
   Array<int> &ess_rtdof_list) const
{

}

void BlockStaticCondensation::ComputeSolution(const Vector &b,
                                              const Vector &sc_sol,
                                              Vector &sol) const
{

}

BlockStaticCondensation::~BlockStaticCondensation()
{

}

}

