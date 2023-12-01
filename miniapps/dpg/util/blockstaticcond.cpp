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

#include "blockstaticcond.hpp"

namespace mfem
{

BlockStaticCondensation::BlockStaticCondensation(Array<FiniteElementSpace *> &
                                                 fes_)
{
   SetSpaces(fes_);

   Array<int> rvdofs;
   Array<int> vdofs;
   Array<int> rdof_edof0;
   for (int k = 0; k<nblocks; k++)
   {
      if (!tr_fes[k]) { continue; }
      rdof_edof0.SetSize(tr_fes[k]->GetVSize());
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         fes[k]->GetElementVDofs(i, vdofs);
         tr_fes[k]->GetElementVDofs(i, rvdofs);
         const int vdim = fes[k]->GetVDim();
         const int nsd = vdofs.Size()/vdim;
         const int nsrd = rvdofs.Size()/vdim;
         for (int vd = 0; vd < vdim; vd++)
         {
            for (int j = 0; j < nsrd; j++)
            {
               int rvdof = rvdofs[j+nsrd*vd];
               int vdof = vdofs[j+nsd*vd];
               if (rvdof < 0)
               {
                  rvdof = -1-rvdof;
                  vdof = -1-vdof;
               }
               MFEM_ASSERT(vdof >= 0, "incompatible volume and trace FE spaces");
               rdof_edof0[rvdof] = vdof + dof_offsets[k];
            }
         }
      }
      rdof_edof.Append(rdof_edof0);
   }
}

void BlockStaticCondensation::SetSpaces(Array<FiniteElementSpace*> & fes_)
{
#ifdef MFEM_USE_MPI
   ParMesh *pmesh = nullptr;
   parallel = false;
   if (dynamic_cast<ParFiniteElementSpace *>(fes_[0]))
   {
      parallel = true;
   }
#else
   parallel = false;
#endif
   fes=fes_;
   nblocks = fes.Size();
   rblocks = 0;
   tr_fes.SetSize(nblocks);
   mesh = fes[0]->GetMesh();

   IsTraceSpace.SetSize(nblocks);
   const FiniteElementCollection * fec;
   for (int i = 0; i < nblocks; i++)
   {
      fec = fes[i]->FEColl();
      IsTraceSpace[i] =
         (dynamic_cast<const H1_Trace_FECollection*>(fec) ||
          dynamic_cast<const ND_Trace_FECollection*>(fec) ||
          dynamic_cast<const RT_Trace_FECollection*>(fec));
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         pmesh = dynamic_cast<ParMesh *>(mesh);
         tr_fes[i] = (fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) ?
                     nullptr : (IsTraceSpace[i]) ? fes[i] :
                     new ParFiniteElementSpace(pmesh, fec->GetTraceCollection(), fes[i]->GetVDim(),
                                               fes[i]->GetOrdering());
      }
      else
      {
         tr_fes[i] = (fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) ?
                     nullptr : (IsTraceSpace[i]) ? fes[i] :
                     new FiniteElementSpace(mesh, fec->GetTraceCollection(), fes[i]->GetVDim(),
                                            fes[i]->GetOrdering());
      }
#else
      // skip if it's an L2 space (no trace space to construct)
      tr_fes[i] = (fec->GetContType() == FiniteElementCollection::DISCONTINUOUS) ?
                  nullptr : (IsTraceSpace[i]) ? fes[i] :
                  new FiniteElementSpace(mesh, fec->GetTraceCollection(), fes[i]->GetVDim(),
                                         fes[i]->GetOrdering());
#endif
      if (tr_fes[i]) { rblocks++; }
   }
   if (parallel)
   {
      ess_tdofs.SetSize(rblocks);
      for (int i = 0; i<rblocks; i++)
      {
         ess_tdofs[i] = new Array<int>();
      }
   }
   Init();
}

void BlockStaticCondensation::ComputeOffsets()
{
   dof_offsets.SetSize(nblocks+1);
   tdof_offsets.SetSize(nblocks+1);
   dof_offsets[0] = 0;
   tdof_offsets[0] = 0;

   rdof_offsets.SetSize(rblocks+1);
   rtdof_offsets.SetSize(rblocks+1);
   rdof_offsets[0] = 0;
   rtdof_offsets[0] = 0;

   int j=0;
   for (int i =0; i<nblocks; i++)
   {
      dof_offsets[i+1] = fes[i]->GetVSize();
      tdof_offsets[i+1] = fes[i]->GetTrueVSize();
      if (tr_fes[i])
      {
         rdof_offsets[j+1] = tr_fes[i]->GetVSize();
         rtdof_offsets[j+1] = tr_fes[i]->GetTrueVSize();
         j++;
      }
   }
   rdof_offsets.PartialSum();
   rtdof_offsets.PartialSum();
   dof_offsets.PartialSum();
   tdof_offsets.PartialSum();
}


void BlockStaticCondensation::Init()
{
   lmat.SetSize(mesh->GetNE());
   lvec.SetSize(mesh->GetNE());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      lmat[i] = nullptr;
      lvec[i] = nullptr;
   }

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

void BlockStaticCondensation::GetReducedElementIndicesAndOffsets(int el,
                                                                 Array<int> & trace_ldofs,
                                                                 Array<int> & interior_ldofs,
                                                                 Array<int> & offsets) const
{
   int dim = mesh->Dimension();
   offsets.SetSize(tr_fes.Size()+1); offsets = 0;
   Array<int> dofs;
   Array<int> faces, ori;
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   else if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else if (dim == 3)
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   else
   {
      MFEM_ABORT("BlockStaticCondensation::GetReducedElementIndicesAndOffsets: "
                 "dim > 3 not supported");
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


void BlockStaticCondensation::GetReducedElementVDofs(int el,
                                                     Array<int> & rdofs) const
{
   Array<int> faces, ori;
   int dim = mesh->Dimension();
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   else if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else if (dim == 3)
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   else
   {
      MFEM_ABORT("BlockStaticCondensation::GetReducedElementVDofs: "
                 "dim > 3 not supported");
   }
   int numfaces = faces.Size();
   rdofs.SetSize(0);
   int skip = 0;
   for (int i = 0; i<tr_fes.Size(); i++)
   {
      if (!tr_fes[i]) { continue; }
      Array<int> vdofs;
      if (IsTraceSpace[i])
      {
         Array<int> face_vdofs;
         for (int k = 0; k < numfaces; k++)
         {
            int iface = faces[k];
            tr_fes[i]->GetFaceVDofs(iface, face_vdofs);
            vdofs.Append(face_vdofs);
         }
      }
      else
      {
         tr_fes[i]->GetElementVDofs(el, vdofs);
      }
      for (int j=0; j<vdofs.Size(); j++)
      {
         vdofs[j] = (vdofs[j]>=0) ? vdofs[j]+rdof_offsets[skip] :
                    vdofs[j]-rdof_offsets[skip];
      }
      skip++;
      rdofs.Append(vdofs);
   }
}

void BlockStaticCondensation::GetElementVDofs(int el, Array<int> & vdofs) const
{
   Array<int> faces, ori;
   int dim = mesh->Dimension();
   if (dim == 1)
   {
      mesh->GetElementVertices(el, faces);
   }
   else if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else if (dim == 3)
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   else
   {
      MFEM_ABORT("BlockStaticCondensation::GetElementVDofs: "
                 "dim > 3 not supported");
   }
   int numfaces = faces.Size();
   vdofs.SetSize(0);
   for (int i = 0; i<tr_fes.Size(); i++)
   {
      Array<int> dofs;
      if (IsTraceSpace[i])
      {
         Array<int> face_vdofs;
         for (int k = 0; k < numfaces; k++)
         {
            int iface = faces[k];
            fes[i]->GetFaceVDofs(iface, face_vdofs);
            dofs.Append(face_vdofs);
         }
      }
      else
      {
         fes[i]->GetElementVDofs(el, dofs);
      }
      for (int j=0; j<dofs.Size(); j++)
      {
         dofs[j] = (dofs[j]>=0) ? dofs[j]+dof_offsets[i] :
                   dofs[j]-dof_offsets[i];
      }
      vdofs.Append(dofs);
   }
}


void BlockStaticCondensation::GetLocalSchurComplement(int el,
                                                      const Array<int> & tr_idx,
                                                      const Array<int> & int_idx,
                                                      const DenseMatrix & elmat,
                                                      const Vector & elvect,
                                                      DenseMatrix & rmat,
                                                      Vector & rvect)
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

   lu.Mult(A_it,*lmat[el]);
   lu.Mult(y_i,*lvec[el]);

   // LHS
   mfem::Mult(A_ti,*lmat[el],rmat);

   rmat.Neg();
   rmat.Add(1., A_tt);

   // RHS
   A_ti.Mult(*lvec[el], rvect);
   rvect.Neg();
   rvect.Add(1., y_t);
}


void BlockStaticCondensation::AssembleReducedSystem(int el,
                                                    DenseMatrix &elmat,
                                                    Vector & elvect)
{
   // Get Schur Complement
   Array<int> tr_idx, int_idx;
   Array<int> offsets;
   // Get local element idx and offsets for global assembly
   GetReducedElementIndicesAndOffsets(el, tr_idx,int_idx, offsets);

   DenseMatrix rmat, *rmatptr;
   Vector rvec, *rvecptr;
   // Extract the reduced matrices based on tr_idx and int_idx
   if (int_idx.Size()!=0)
   {
      GetLocalSchurComplement(el,tr_idx,int_idx, elmat, elvect, rmat, rvec);
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
   else if (dim == 2)
   {
      mesh->GetElementEdges(el, faces, ori);
   }
   else if (dim == 3)
   {
      mesh->GetElementFaces(el,faces,ori);
   }
   else
   {
      MFEM_ABORT("BlockStaticCondensation::AssembleReducedSystem: "
                 "dim > 3 not supported");
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

#ifdef MFEM_USE_MPI
void BlockStaticCondensation::BuildParallelProlongation()
{
   MFEM_VERIFY(parallel, "BuildParallelProlongation: wrong code path");
   pP = new BlockOperator(rdof_offsets, rtdof_offsets);
   R = new BlockMatrix(rtdof_offsets, rdof_offsets);
   pP->owns_blocks = 0;
   R->owns_blocks = 0;
   int skip = 0;
   for (int i = 0; i<nblocks; i++)
   {
      if (!tr_fes[i]) { continue; }
      const HypreParMatrix *P_ =
         dynamic_cast<ParFiniteElementSpace *>(tr_fes[i])->Dof_TrueDof_Matrix();
      if (P_)
      {
         const SparseMatrix *R_ = tr_fes[i]->GetRestrictionMatrix();
         pP->SetBlock(skip,skip,const_cast<HypreParMatrix*>(P_));
         R->SetBlock(skip,skip,const_cast<SparseMatrix*>(R_));
      }
      skip++;
   }
}

void BlockStaticCondensation::ParallelAssemble(BlockMatrix *m)
{
   if (!pP) { BuildParallelProlongation(); }

   pS = new BlockOperator(rtdof_offsets);
   pS_e = new BlockOperator(rtdof_offsets);
   pS->owns_blocks = 1;
   pS_e->owns_blocks = 1;
   HypreParMatrix * A = nullptr;
   HypreParMatrix * PtAP = nullptr;
   int skip_i=0;
   ParFiniteElementSpace * pfes_i = nullptr;
   ParFiniteElementSpace * pfes_j = nullptr;
   for (int i = 0; i<nblocks; i++)
   {
      if (!tr_fes[i]) { continue; }
      pfes_i = dynamic_cast<ParFiniteElementSpace*>(tr_fes[i]);
      HypreParMatrix * Pi = (HypreParMatrix*)(&pP->GetBlock(skip_i,skip_i));
      int skip_j=0;
      for (int j = 0; j<nblocks; j++)
      {
         if (!tr_fes[j]) { continue; }
         if (m->IsZeroBlock(skip_i,skip_j)) { continue; }
         if (skip_i == skip_j)
         {
            // Make block diagonal square hypre matrix
            A = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                   pfes_i->GetDofOffsets(),&m->GetBlock(skip_i,skip_i));
            PtAP = RAP(A,Pi);
            delete A;
            pS_e->SetBlock(skip_i,skip_i,PtAP->EliminateRowsCols(*ess_tdofs[skip_i]));
         }
         else
         {
            pfes_j = dynamic_cast<ParFiniteElementSpace*>(tr_fes[j]);
            HypreParMatrix * Pj = (HypreParMatrix*)(&pP->GetBlock(skip_j,skip_j));
            A = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                   pfes_j->GlobalVSize(), pfes_i->GetDofOffsets(),
                                   pfes_j->GetDofOffsets(), &m->GetBlock(skip_i,skip_j));
            PtAP = RAP(Pi,A,Pj);
            delete A;
            pS_e->SetBlock(skip_i,skip_j,PtAP->EliminateCols(*ess_tdofs[skip_j]));
            PtAP->EliminateRows(*ess_tdofs[skip_i]);
         }
         pS->SetBlock(skip_i,skip_j,PtAP);
         skip_j++;
      }
      skip_i++;
   }
}

#endif


void BlockStaticCondensation::ConformingAssemble(int skip_zeros)
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA = mfem::Mult(*Pt, *S);
   delete S;
   if (S_e)
   {
      BlockMatrix *PtAe = mfem::Mult(*Pt, *S_e);
      delete S_e;
      S_e = PtAe;
   }
   delete Pt;
   S = mfem::Mult(*PtA, *P);
   delete PtA;

   if (S_e)
   {
      BlockMatrix *PtAeP = mfem::Mult(*S_e, *P);
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
   if (!parallel)
   {
      if (!S_e)
      {
         bool conforming = true;
         for (int i = 0; i<nblocks; i++)
         {
            if (!tr_fes[i]) { continue; }
            const SparseMatrix *P_ = tr_fes[i]->GetConformingProlongation();
            if (P_)
            {
               conforming = false;
               break;
            }
         }
         if (!conforming) { ConformingAssemble(0); }
         const int remove_zeros = 0;
         EliminateReducedTrueDofs(ess_rtdof_list, diag_policy);
         Finalize(remove_zeros);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      FillEssTdofLists(ess_rtdof_list);
      if (S)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         ParallelAssemble(S);
         delete S;   S=nullptr;
         delete S_e; S_e = nullptr;
      }
#endif
   }
}


void BlockStaticCondensation::ConvertMarkerToReducedTrueDofs(
   Array<int> & tdof_marker,
   Array<int> & rtdof_marker)
{
   // convert tdof_marker to dof_marker
   rtdof_marker.SetSize(0);
   Array<int> tdof_marker0;
   Array<int> dof_marker0;
   Array<int> dof_marker;
   int * data = tdof_marker.GetData();
   for (int i = 0; i<nblocks; i++)
   {
      tdof_marker0.MakeRef(&data[tdof_offsets[i]],tdof_offsets[i+1]-tdof_offsets[i]);
      const SparseMatrix * R_ = fes[i]->GetRestrictionMatrix();
      if (!R_)
      {
         dof_marker0.MakeRef(tdof_marker0);
      }
      else
      {
         dof_marker0.SetSize(fes[i]->GetVSize());
         R_->BooleanMultTranspose(tdof_marker0, dof_marker0);
      }
      dof_marker.Append(dof_marker0);
   }

   int rdofs = rdof_edof.Size();
   Array<int> rdof_marker(rdofs);

   for (int i = 0; i < rdofs; i++)
   {
      rdof_marker[i] = dof_marker[rdof_edof[i]];
   }

   // convert rdof_marker to rtdof_marker
   Array<int> rtdof_marker0;
   Array<int> rdof_marker0;
   int * rdata = rdof_marker.GetData();
   int k=0;
   for (int i = 0; i<nblocks; i++)
   {
      if (!tr_fes[i]) { continue; }
      rdof_marker0.MakeRef(&rdata[rdof_offsets[k]],rdof_offsets[k+1]-rdof_offsets[k]);
      const SparseMatrix *tr_R = tr_fes[i]->GetRestrictionMatrix();
      if (!tr_R)
      {
         rtdof_marker0.MakeRef(rdof_marker0);
      }
      else
      {
         rtdof_marker0.SetSize(tr_fes[i]->GetTrueVSize());
         tr_R->BooleanMult(rdof_marker0, rtdof_marker0);
      }
      rtdof_marker.Append(rtdof_marker0);
      k++;
   }
}

void BlockStaticCondensation::FillEssTdofLists(const Array<int> & ess_tdof_list)
{
   int j;
   for (int i = 0; i<ess_tdof_list.Size(); i++)
   {
      int tdof = ess_tdof_list[i];
      for (j = 0; j < rblocks; j++)
      {
         if (rtdof_offsets[j+1] > tdof) { break; }
      }
      ess_tdofs[j]->Append(tdof-rtdof_offsets[j]);
   }
}

void BlockStaticCondensation::SetEssentialTrueDofs(const Array<int>
                                                   &ess_tdof_list)
{
   Array<int> tdof_marker;
   Array<int> rtdof_marker;
   FiniteElementSpace::ListToMarker(ess_tdof_list,tdof_offsets.Last(),tdof_marker);
   ConvertMarkerToReducedTrueDofs(tdof_marker, rtdof_marker);
   FiniteElementSpace::MarkerToList(rtdof_marker,ess_rtdof_list);
}

void BlockStaticCondensation::EliminateReducedTrueDofs(const Array<int>
                                                       &ess_rtdof_list_,
                                                       Matrix::DiagonalPolicy dpolicy)
{
   MFEM_VERIFY(!parallel, "EliminateReducedTrueDofs::Wrong Code path");

   if (S_e == NULL)
   {
      Array<int> offsets;

      offsets.MakeRef( (P) ? rtdof_offsets : rdof_offsets);

      S_e = new BlockMatrix(offsets);
      S_e->owns_blocks = 1;
      for (int i = 0; i<S_e->NumRowBlocks(); i++)
      {
         int h = offsets[i+1] - offsets[i];
         for (int j = 0; j<S_e->NumColBlocks(); j++)
         {
            int w = offsets[j+1] - offsets[j];
            S_e->SetBlock(i,j,new SparseMatrix(h, w));
         }
      }
   }
   S->EliminateRowCols(ess_rtdof_list_,S_e,dpolicy);
}

void BlockStaticCondensation::ReduceSolution(const Vector &sol,
                                             Vector &sc_sol) const
{
   MFEM_ASSERT(sol.Size() == dof_offsets.Last(), "'sol' has incorrect size");
   const int nrdofs = rdof_offsets.Last();
   Vector sol_r;
   if (!R)
   {
      sc_sol.SetSize(nrdofs);
      sol_r.SetDataAndSize(sc_sol.GetData(), sc_sol.Size());
   }
   else
   {
      sol_r.SetSize(nrdofs);
   }
   for (int i = 0; i < nrdofs; i++)
   {
      sol_r(i) = sol(rdof_edof[i]);
   }
   if (R)
   {
      // wrap vector into a block vector
      BlockVector blsol_r(sol_r,rdof_offsets);
      sc_sol.SetSize(R->Height());
      R->Mult(blsol_r, sc_sol);
   }
}

void BlockStaticCondensation::ReduceSystem(Vector &x, Vector &X,
                                           Vector &B,
                                           int copy_interior) const
{
   ReduceSolution(x, X);
   if (!parallel)
   {
      if (!P)
      {
         S_e->AddMult(X,*y,-1.);
         S->PartMult(ess_rtdof_list,X,*y);
         B.MakeRef(*y, 0, y->Size());
      }
      else
      {
         B.SetSize(P->Width());
         P->MultTranspose(*y, B);
         S_e->AddMult(X,B,-1.);
         S->PartMult(ess_rtdof_list,X,B);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      B.SetSize(pP->Width());
      pP->MultTranspose(*y,B);

      Vector tmp(B.Size());
      pS_e->Mult(X,tmp);
      B-=tmp;
      for (int j = 0; j<rblocks; j++)
      {
         if (!ess_tdofs[j]->Size()) { continue; }
         HypreParMatrix *Ah = (HypreParMatrix *)(&pS->GetBlock(j,j));
         Vector diag;
         Ah->GetDiag(diag);
         for (int i = 0; i < ess_tdofs[j]->Size(); i++)
         {
            int tdof = (*ess_tdofs[j])[i];
            int gdof = tdof + rtdof_offsets[j];
            B(gdof) = diag(tdof)*X(gdof);
         }
      }
#endif
   }
   if (!copy_interior) { X.SetSubVectorComplement(ess_rtdof_list, 0.0); }
}


void BlockStaticCondensation::ComputeSolution(const Vector &sc_sol,
                                              Vector &sol) const
{

   const int nrdofs = rdof_offsets.Last();
   const int nrtdofs = rtdof_offsets.Last();
   MFEM_VERIFY(sc_sol.Size() == nrtdofs, "'sc_sol' has incorrect size");

   Vector sol_r;
   if (!parallel)
   {
      if (!P)
      {
         sol_r.SetDataAndSize(sc_sol.GetData(), sc_sol.Size());
      }
      else
      {
         sol_r.SetSize(nrdofs);
         P->Mult(sc_sol, sol_r);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      sol_r.SetSize(nrdofs);
      pP->Mult(sc_sol, sol_r);
#endif
   }

   if (rdof_offsets.Last() == dof_offsets.Last())
   {
      sol = sol_r;
      return;
   }
   else
   {
      sol.SetSize(dof_offsets.Last());
   }

   Vector lsr; // element (local) sc solution vector
   Vector lsi; // element (local) interior solution vector
   const int NE = mesh->GetNE();

   Array<int> trace_vdofs;
   Array<int> vdofs;
   Array<int> tr_offsets;
   Vector lsol;
   for (int iel = 0; iel < NE; iel++)
   {
      lsol.SetSize(lmat[iel]->Width() + lmat[iel]->Height());
      GetReducedElementVDofs(iel, trace_vdofs);

      lsr.SetSize(trace_vdofs.Size());
      sol_r.GetSubVector(trace_vdofs, lsr);

      // complete the interior dofs
      lsi.SetSize(lmat[iel]->Height());
      lmat[iel]->Mult(lsr,lsi);
      lsi.Neg();
      lsi+=*lvec[iel];

      Array<int> tr_idx,int_idx,idx_offs;
      GetReducedElementIndicesAndOffsets(iel,tr_idx, int_idx, idx_offs);
      lsol.SetSubVector(tr_idx,lsr);

      lsol.SetSubVector(int_idx,lsi);

      GetElementVDofs(iel, vdofs);
      sol.SetSubVector(vdofs,lsol);

   }

}

BlockStaticCondensation::~BlockStaticCondensation()
{
   delete S_e; S_e = nullptr;
   delete S; S=nullptr;
   delete y; y=nullptr;

   if (P) { delete P; } P=nullptr;
   if (R) { delete R; } R=nullptr;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      delete pS; pS=nullptr;
      delete pS_e; pS_e=nullptr;
      for (int i = 0; i<rblocks; i++)
      {
         delete ess_tdofs[i];
      }
      delete pP; pP=nullptr;
   }
#endif

   for (int i=0; i<lmat.Size(); i++)
   {
      delete lmat[i]; lmat[i] = nullptr;
      delete lvec[i]; lvec[i] = nullptr;
   }
}

} // namespace mfem
