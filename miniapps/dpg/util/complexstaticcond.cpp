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

#include "complexstaticcond.hpp"

namespace mfem
{

ComplexBlockStaticCondensation::ComplexBlockStaticCondensation(
   Array<FiniteElementSpace *> &
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

void ComplexBlockStaticCondensation::SetSpaces(Array<FiniteElementSpace*> &
                                               fes_)
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

void ComplexBlockStaticCondensation::ComputeOffsets()
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

void ComplexBlockStaticCondensation::Init()
{
   lmat.SetSize(mesh->GetNE());
   lvec.SetSize(mesh->GetNE());
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      lmat[i] = nullptr;
      lvec[i] = nullptr;
   }

   ComputeOffsets();

   S_r = new BlockMatrix(rdof_offsets);
   S_r->owns_blocks = 1;
   S_i = new BlockMatrix(rdof_offsets);
   S_i->owns_blocks = 1;

   for (int i = 0; i<S_r->NumRowBlocks(); i++)
   {
      int h = rdof_offsets[i+1] - rdof_offsets[i];
      for (int j = 0; j<S_r->NumColBlocks(); j++)
      {
         int w = rdof_offsets[j+1] - rdof_offsets[j];
         S_r->SetBlock(i,j,new SparseMatrix(h, w));
         S_i->SetBlock(i,j,new SparseMatrix(h, w));
      }
   }

   y = new Vector(2*rdof_offsets.Last());
   *y=0.;
   y_r = new BlockVector(*y, rdof_offsets);
   y_i = new BlockVector(*y, rdof_offsets.Last(), rdof_offsets);
}

void ComplexBlockStaticCondensation::GetReduceElementIndicesAndOffsets(int el,
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
      MFEM_ABORT("ComplexBlockStaticCondensation::GetReduceElementIndicesAndOffsets: "
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


void ComplexBlockStaticCondensation::GetReduceElementVDofs(int el,
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
      MFEM_ABORT("ComplexBlockStaticCondensation::GetReduceElementVDofs: "
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
void ComplexBlockStaticCondensation::GetElementVDofs(int el,
                                                     Array<int> & vdofs) const
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
      MFEM_ABORT("ComplexBlockStaticCondensation::GetElementVDofs: "
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


ComplexDenseMatrix * ComplexBlockStaticCondensation::GetLocalShurComplement(
   int el,
   const Array<int> & tr_idx, const Array<int> & int_idx,
   const ComplexDenseMatrix & elmat,
   const Vector & elvect_real, const Vector & elvect_imag,
   Vector & rvect_real, Vector & rvect_imag)
{
   int rdofs = tr_idx.Size();
   int idofs = int_idx.Size();
   MFEM_VERIFY(idofs != 0, "Number of interior dofs is zero");
   MFEM_VERIFY(rdofs != 0, "Number of interface dofs is zero");

   DenseMatrix A_tt_real, A_ti_real, A_it_real, A_ii_real;
   DenseMatrix A_tt_imag, A_ti_imag, A_it_imag, A_ii_imag;


   Vector yt(2*rdofs);
   Vector yi(2*idofs);

   Vector yt_real(yt, 0,rdofs);
   Vector yt_imag(yt, rdofs, rdofs);

   Vector yi_real(yi, 0, idofs);
   Vector yi_imag(yi,idofs, idofs);

   // real part of Matrix and vectors
   elmat.real().GetSubMatrix(tr_idx,A_tt_real);
   elmat.real().GetSubMatrix(tr_idx,int_idx, A_ti_real);
   elmat.real().GetSubMatrix(int_idx, tr_idx, A_it_real);
   elmat.real().GetSubMatrix(int_idx, A_ii_real);

   elvect_real.GetSubVector(tr_idx, yt_real);
   elvect_real.GetSubVector(int_idx, yi_real);

   // imag part of Matrix and vectors
   elmat.imag().GetSubMatrix(tr_idx,A_tt_imag);
   elmat.imag().GetSubMatrix(tr_idx,int_idx, A_ti_imag);
   elmat.imag().GetSubMatrix(int_idx, tr_idx, A_it_imag);
   elmat.imag().GetSubMatrix(int_idx, A_ii_imag);

   elvect_imag.GetSubVector(tr_idx, yt_imag);
   elvect_imag.GetSubVector(int_idx, yi_imag);

   // construct complex
   ComplexDenseMatrix A_tt(&A_tt_real,&A_tt_imag,false,false);
   ComplexDenseMatrix A_ti(&A_ti_real,&A_ti_imag,false,false);
   ComplexDenseMatrix A_it(&A_it_real,&A_it_imag,false,false);
   ComplexDenseMatrix A_ii(&A_ii_real,&A_ii_imag,false,false);

   ComplexDenseMatrix * invA_ii = A_ii.ComputeInverse();

   // LHS
   lmat[el] = mfem::Mult(*invA_ii,A_it);
   ComplexDenseMatrix * rmat = mfem::Mult(A_ti,*lmat[el]);
   rmat->real().Neg();
   rmat->imag().Neg();
   rmat->real().Add(1., A_tt.real());
   rmat->imag().Add(1., A_tt.imag());

   // RHS
   lvec[el] = new Vector(2*idofs);
   invA_ii->Mult(yi,*lvec[el]);
   delete invA_ii;

   Vector rvect(2*rdofs);
   A_ti.Mult(*lvec[el], rvect);
   rvect_real.SetSize(rdofs);
   rvect_imag.SetSize(rdofs);
   for (int i = 0; i<rdofs; i++)
   {
      rvect_real(i) = yt_real(i) - rvect(i);
      rvect_imag(i) = yt_imag(i) - rvect(i+rdofs);
   }
   return rmat;
}


void ComplexBlockStaticCondensation::AssembleReducedSystem(int el,
                                                           ComplexDenseMatrix &elmat,
                                                           Vector & elvect_r, Vector & elvect_i)
{
   // Get Shur Complement
   Array<int> tr_idx, int_idx;
   Array<int> offsets;
   // Get local element idx and offsets for global assembly
   GetReduceElementIndicesAndOffsets(el, tr_idx,int_idx, offsets);

   ComplexDenseMatrix *rmat = nullptr;
   Vector rvec_real, *rvecptr_real;
   Vector rvec_imag, *rvecptr_imag;
   // Extract the reduced matrices based on tr_idx and int_idx
   if (int_idx.Size()!=0)
   {
      rmat = GetLocalShurComplement(el,tr_idx,int_idx, elmat, elvect_r, elvect_i,
                                    rvec_real,rvec_imag);
      rvecptr_real = &rvec_real;
      rvecptr_imag = &rvec_imag;
   }
   else
   {
      rmat = &elmat;
      rvecptr_real = &elvect_r;
      rvecptr_imag = &elvect_i;
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
      MFEM_ABORT("ComplexBlockStaticCondensation::AssembleReducedSystem: "
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

         DenseMatrix Ae_r, Ae_i;
         rmat->real().GetSubMatrix(offsets[i],offsets[i+1],
                                   offsets[j],offsets[j+1], Ae_r);
         rmat->imag().GetSubMatrix(offsets[i],offsets[i+1],
                                   offsets[j],offsets[j+1], Ae_i);
         if (doftrans_i || doftrans_j)
         {
            TransformDual(doftrans_i, doftrans_j, Ae_r);
            TransformDual(doftrans_i, doftrans_j, Ae_i);
         }
         S_r->GetBlock(skip_i,skip_j).AddSubMatrix(vdofs_i,vdofs_j, Ae_r);
         S_i->GetBlock(skip_i,skip_j).AddSubMatrix(vdofs_i,vdofs_j, Ae_i);
         skip_j++;
      }

      // assemble rhs
      Vector vec1_r(*rvecptr_real, offsets[i], offsets[i+1]-offsets[i]);
      Vector vec1_i(*rvecptr_imag, offsets[i], offsets[i+1]-offsets[i]);
      // ref subvector
      if (doftrans_i)
      {
         doftrans_i->TransformDual(vec1_r);
         doftrans_i->TransformDual(vec1_i);
      }
      y_r->GetBlock(skip_i).AddElementVector(vdofs_i,vec1_r);
      y_i->GetBlock(skip_i).AddElementVector(vdofs_i,vec1_i);
      skip_i++;
   }
   if (int_idx.Size()!=0) { delete rmat; }
}

void ComplexBlockStaticCondensation::BuildProlongation()
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
void ComplexBlockStaticCondensation::BuildParallelProlongation()
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

void ComplexBlockStaticCondensation::ParallelAssemble(BlockMatrix *m_r,
                                                      BlockMatrix *m_i)
{

   if (!pP) { BuildParallelProlongation(); }

   pS_r = new BlockOperator(rtdof_offsets);
   pS_e_r = new BlockOperator(rtdof_offsets);
   pS_i = new BlockOperator(rtdof_offsets);
   pS_e_i = new BlockOperator(rtdof_offsets);
   pS_r->owns_blocks = 1;
   pS_i->owns_blocks = 1;
   pS_e_r->owns_blocks = 1;
   pS_e_i->owns_blocks = 1;
   HypreParMatrix * A_r = nullptr;
   HypreParMatrix * A_i = nullptr;
   HypreParMatrix * PtAP_r = nullptr;
   HypreParMatrix * PtAP_i = nullptr;
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
         if (m_r->IsZeroBlock(skip_i,skip_j)) { continue; }
         if (skip_i == skip_j)
         {
            // Make block diagonal square hypre matrix
            A_r = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                     pfes_i->GetDofOffsets(),&m_r->GetBlock(skip_i,skip_i));
            PtAP_r = RAP(A_r,Pi);
            delete A_r;

            pS_e_r->SetBlock(skip_i,skip_i,PtAP_r->EliminateRowsCols(*ess_tdofs[skip_i]));

            A_i = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                     pfes_i->GetDofOffsets(),&m_i->GetBlock(skip_i,skip_i));
            PtAP_i = RAP(A_i,Pi);
            delete A_i;
            pS_e_i->SetBlock(skip_i,skip_j,PtAP_i->EliminateCols(*ess_tdofs[skip_j]));
            PtAP_i->EliminateRows(*ess_tdofs[skip_i]);
         }
         else
         {
            pfes_j = dynamic_cast<ParFiniteElementSpace*>(tr_fes[j]);
            HypreParMatrix * Pj = (HypreParMatrix*)(&pP->GetBlock(skip_j,skip_j));
            A_r = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                     pfes_j->GlobalVSize(), pfes_i->GetDofOffsets(),
                                     pfes_j->GetDofOffsets(), &m_r->GetBlock(skip_i,skip_j));
            PtAP_r = RAP(Pi,A_r,Pj);
            delete A_r;
            pS_e_r->SetBlock(skip_i,skip_j,PtAP_r->EliminateCols(*ess_tdofs[skip_j]));
            PtAP_r->EliminateRows(*ess_tdofs[skip_i]);

            A_i = new HypreParMatrix(pfes_i->GetComm(), pfes_i->GlobalVSize(),
                                     pfes_j->GlobalVSize(), pfes_i->GetDofOffsets(),
                                     pfes_j->GetDofOffsets(), &m_i->GetBlock(skip_i,skip_j));
            PtAP_i = RAP(Pi,A_i,Pj);
            delete A_i;
            pS_e_i->SetBlock(skip_i,skip_j,PtAP_i->EliminateCols(*ess_tdofs[skip_j]));
            PtAP_i->EliminateRows(*ess_tdofs[skip_i]);

         }
         pS_r->SetBlock(skip_i,skip_j,PtAP_r);
         pS_i->SetBlock(skip_i,skip_j,PtAP_i);
         skip_j++;
      }
      skip_i++;
   }
}

#endif


void ComplexBlockStaticCondensation::ConformingAssemble(int skip_zeros)
{
   Finalize(0);
   if (!P) { BuildProlongation(); }

   BlockMatrix * Pt = Transpose(*P);
   BlockMatrix * PtA_r = mfem::Mult(*Pt, *S_r);
   BlockMatrix * PtA_i = mfem::Mult(*Pt, *S_i);
   delete S_r;
   delete S_i;
   if (S_e_r)
   {
      BlockMatrix *PtAe_r = mfem::Mult(*Pt, *S_e_r);
      BlockMatrix *PtAe_i = mfem::Mult(*Pt, *S_e_i);
      delete S_e_r;
      delete S_e_i;
      S_e_r = PtAe_r;
      S_e_i = PtAe_i;
   }
   delete Pt;
   S_r = mfem::Mult(*PtA_r, *P);
   S_i = mfem::Mult(*PtA_i, *P);
   delete PtA_r;
   delete PtA_i;

   if (S_e_r)
   {
      BlockMatrix *PtAeP_r = mfem::Mult(*S_e_r, *P);
      BlockMatrix *PtAeP_i = mfem::Mult(*S_e_i, *P);
      S_e_r = PtAeP_r;
      S_e_i = PtAeP_i;
   }
   height = 2*S_r->Height();
   width = 2*S_r->Width();
}

void ComplexBlockStaticCondensation::Finalize(int skip_zeros)
{
   if (S_r)
   {
      S_r->Finalize(skip_zeros);
      S_i->Finalize(skip_zeros);
   }
   if (S_e_r)
   {
      S_e_r->Finalize(skip_zeros);
      S_e_i->Finalize(skip_zeros);
   }
}

void ComplexBlockStaticCondensation::FormSystemMatrix(Operator::DiagonalPolicy
                                                      diag_policy)
{

   if (!parallel)
   {
      if (!S_e_r)
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
      if (S_r)
      {
         const int remove_zeros = 0;
         Finalize(remove_zeros);
         ParallelAssemble(S_r, S_i);
         delete S_r;  S_r=nullptr;
         delete S_i;  S_i=nullptr;
         delete S_e_r; S_e_r = nullptr;
         delete S_e_i; S_e_i = nullptr;
      }
#endif
   }
}

void ComplexBlockStaticCondensation::ConvertMarkerToReducedTrueDofs(
   Array<int> & tdof_marker,
   Array<int> & rtdof_marker)
{
   // convert tdof_marker to dof_marker
   rtdof_marker.SetSize(0);
   Array<int> tdof_marker0;
   Array<int> dof_marker0;
   Array<int> dof_marker;

   for (int i = 0; i<nblocks; i++)
   {
      tdof_marker0.MakeRef(&tdof_marker[tdof_offsets[i]],
                           tdof_offsets[i+1]-tdof_offsets[i]);
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
   int k=0;
   for (int i = 0; i<nblocks; i++)
   {
      if (!tr_fes[i]) { continue; }
      rdof_marker0.MakeRef(&rdof_marker[rdof_offsets[k]],
                           rdof_offsets[k+1]-rdof_offsets[k]);
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

void ComplexBlockStaticCondensation::FillEssTdofLists(const Array<int> &
                                                      ess_tdof_list)
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

void ComplexBlockStaticCondensation::SetEssentialTrueDofs(const Array<int>
                                                          &ess_tdof_list)
{
   Array<int> tdof_marker;
   Array<int> rtdof_marker;
   FiniteElementSpace::ListToMarker(ess_tdof_list,tdof_offsets.Last(),tdof_marker);
   ConvertMarkerToReducedTrueDofs(tdof_marker, rtdof_marker);
   FiniteElementSpace::MarkerToList(rtdof_marker,ess_rtdof_list);
}

void ComplexBlockStaticCondensation::EliminateReducedTrueDofs(const Array<int>
                                                              &ess_rtdof_list_,
                                                              Matrix::DiagonalPolicy dpolicy)
{

   MFEM_VERIFY(!parallel, "EliminateReducedTrueDofs::Wrong code path");

   if (S_e_r == NULL)
   {
      Array<int> offsets;

      offsets.MakeRef( (P) ? rtdof_offsets : rdof_offsets);

      S_e_r = new BlockMatrix(offsets);
      S_e_i = new BlockMatrix(offsets);
      S_e_r->owns_blocks = 1;
      S_e_i->owns_blocks = 1;
      for (int i = 0; i<S_e_r->NumRowBlocks(); i++)
      {
         int h = offsets[i+1] - offsets[i];
         for (int j = 0; j<S_e_r->NumColBlocks(); j++)
         {
            int w = offsets[j+1] - offsets[j];
            S_e_r->SetBlock(i,j,new SparseMatrix(h, w));
            S_e_i->SetBlock(i,j,new SparseMatrix(h, w));
         }
      }
   }
   S_r->EliminateRowCols(ess_rtdof_list_,S_e_r,dpolicy);
   S_i->EliminateRowCols(ess_rtdof_list_,S_e_i,
                         Operator::DiagonalPolicy::DIAG_ZERO);
}

void ComplexBlockStaticCondensation::ReduceSolution(const Vector &sol,
                                                    Vector &sc_sol) const
{
   MFEM_ASSERT(sol.Size() == 2*dof_offsets.Last(), "'sol' has incorrect size");
   const int nrdofs = rdof_offsets.Last();

   Vector sol_r_real;
   Vector sol_r_imag;

   if (!R)
   {
      sc_sol.SetSize(2*nrdofs);
      sol_r_real.MakeRef(sc_sol, 0,  nrdofs);
      sol_r_imag.MakeRef(sc_sol, nrdofs, nrdofs);
   }
   else
   {
      sol_r_real.SetSize(nrdofs);
      sol_r_imag.SetSize(nrdofs);
   }
   for (int i = 0; i < nrdofs; i++)
   {
      sol_r_real(i) = sol(rdof_edof[i]);
      sol_r_imag(i) = sol(rdof_edof[i] + dof_offsets.Last());
   }

   if (R)
   {
      int n = R->Height();
      sc_sol.SetSize(2*n);
      Vector sc_real(sc_sol, 0, n);
      Vector sc_imag(sc_sol, n, n);

      // wrap vector into a block vector
      BlockVector blsol_r_real(sol_r_real,rdof_offsets);
      BlockVector blsol_r_imag(sol_r_imag,rdof_offsets);
      R->Mult(blsol_r_real, sc_real);
      R->Mult(blsol_r_imag, sc_imag);
   }
}

void ComplexBlockStaticCondensation::ReduceSystem(Vector &x, Vector &X,
                                                  Vector &B,
                                                  int copy_interior) const
{
   ReduceSolution(x, X);
   Vector X_r(X,0, X.Size()/2);
   Vector X_i(X, X.Size()/2, X.Size()/2);

   if (!parallel)
   {
      if (!P)
      {

         S_e_r->AddMult(X_r,*y_r,-1.);
         S_e_i->AddMult(X_i,*y_r,1.);
         S_e_r->AddMult(X_i,*y_i,-1.);
         S_e_i->AddMult(X_r,*y_i,-1.);

         S_r->PartMult(ess_rtdof_list,X_r,*y_r);
         S_r->PartMult(ess_rtdof_list,X_i,*y_i);
         B.MakeRef(*y, 0, y->Size());
      }
      else
      {
         B.SetSize(2*P->Width());
         Vector B_r(B, 0, P->Width());
         Vector B_i(B, P->Width(), P->Width());

         P->MultTranspose(*y_r, B_r);
         P->MultTranspose(*y_i, B_i);

         S_e_r->AddMult(X_r,B_r,-1.);
         S_e_i->AddMult(X_i,B_r,1.);
         S_e_r->AddMult(X_i,B_i,-1.);
         S_e_i->AddMult(X_r,B_i,-1.);
         S_r->PartMult(ess_rtdof_list,X_r,B_r);
         S_r->PartMult(ess_rtdof_list,X_i,B_i);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      int n = pP->Width();
      B.SetSize(2*n);
      Vector B_r(B, 0, n);
      Vector B_i(B, n, n);

      pP->MultTranspose(*y_r,B_r);
      pP->MultTranspose(*y_i,B_i);

      Vector tmp(B_r.Size());
      pS_e_r->Mult(X_r,tmp); B_r-=tmp;
      pS_e_i->Mult(X_i,tmp); B_r+=tmp;

      pS_e_i->Mult(X_r,tmp); B_i-=tmp;
      pS_e_r->Mult(X_i,tmp); B_i-=tmp;

      for (int j = 0; j<rblocks; j++)
      {
         if (!ess_tdofs[j]->Size()) { continue; }
         for (int i = 0; i < ess_tdofs[j]->Size(); i++)
         {
            int tdof = (*ess_tdofs[j])[i];
            int gdof = tdof + rtdof_offsets[j];
            B_r(gdof) = X_r(gdof);
            B_i(gdof) = X_i(gdof);
         }
      }
#endif
   }
   if (!copy_interior)
   {
      X_r.SetSubVectorComplement(ess_rtdof_list, 0.0);
      X_i.SetSubVectorComplement(ess_rtdof_list, 0.0);
   }
}


void ComplexBlockStaticCondensation::ComputeSolution(const Vector &sc_sol,
                                                     Vector &sol) const
{

   const int nrdofs = rdof_offsets.Last();
   const int nrtdofs = rtdof_offsets.Last();
   MFEM_VERIFY(sc_sol.Size() == 2*nrtdofs, "'sc_sol' has incorrect size");

   Vector sol_r_real;
   Vector sol_r_imag;
   if (!parallel)
   {
      if (!P)
      {
         sol_r_real.MakeRef(const_cast<Vector &>(sc_sol), 0, sc_sol.Size()/2);
         sol_r_imag.MakeRef(const_cast<Vector &>(sc_sol), sc_sol.Size()/2,
                            sc_sol.Size()/2);
      }
      else
      {
         Vector sc_real(const_cast<Vector &>(sc_sol),0, nrtdofs);
         Vector sc_imag(const_cast<Vector &>(sc_sol),nrtdofs, nrtdofs);
         sol_r_real.SetSize(nrdofs);
         sol_r_imag.SetSize(nrdofs);
         P->Mult(sc_real, sol_r_real);
         P->Mult(sc_imag, sol_r_imag);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      Vector sc_real(const_cast<Vector &>(sc_sol),0, nrtdofs);
      Vector sc_imag(const_cast<Vector &>(sc_sol),nrtdofs, nrtdofs);
      sol_r_real.SetSize(nrdofs);
      sol_r_imag.SetSize(nrdofs);
      pP->Mult(sc_real, sol_r_real);
      pP->Mult(sc_imag, sol_r_imag);
#endif
   }

   sol.SetSize(2*dof_offsets.Last());
   Vector sol_real(sol,0,dof_offsets.Last());
   Vector sol_imag(sol,dof_offsets.Last(),dof_offsets.Last());

   if (rdof_offsets.Last() == dof_offsets.Last())
   {
      sol_real = sol_r_real;
      sol_imag = sol_r_imag;
      return;
   }

   Vector lsr; // element (local) sc solution vector
   Vector lsr_real; // element (local) sc solution vector
   Vector lsr_imag; // element (local) sc solution vector
   Vector lsi; // element (local) interior solution vector
   Vector lsi_real; // element (local) interior solution vector
   Vector lsi_imag; // element (local) interior solution vector

   const int NE = mesh->GetNE();

   Array<int> trace_vdofs;
   Array<int> vdofs;
   Array<int> tr_offsets;
   Vector lsol;
   Vector lsol_real;
   Vector lsol_imag;
   for (int iel = 0; iel < NE; iel++)
   {
      GetReduceElementVDofs(iel, trace_vdofs);

      int n = trace_vdofs.Size();
      lsr.SetSize(2*n);
      lsr_real.MakeRef(lsr, 0, n);
      lsr_imag.MakeRef(lsr, n, n);
      sol_r_real.GetSubVector(trace_vdofs, lsr_real);
      sol_r_imag.GetSubVector(trace_vdofs, lsr_imag);

      // complete the interior dofs
      int m = lmat[iel]->Height()/2;
      lsi.SetSize(2*m);
      lsi_real.MakeRef(lsi, 0, m);
      lsi_imag.MakeRef(lsi, m, m);
      lmat[iel]->Mult(lsr,lsi);
      lsi.Neg();
      lsi+=*lvec[iel];

      Array<int> tr_idx,int_idx,idx_offs;
      GetReduceElementIndicesAndOffsets(iel,tr_idx, int_idx, idx_offs);

      // complete all the dofs in the element
      int k = (lmat[iel]->Width() + lmat[iel]->Height())/2;
      lsol.SetSize(2*k);
      lsol_real.MakeRef(lsol, 0, k);
      lsol_imag.MakeRef(lsol, k, k);

      lsol_real.SetSubVector(tr_idx,lsr_real);
      lsol_real.SetSubVector(int_idx,lsi_real);
      lsol_imag.SetSubVector(tr_idx,lsr_imag);
      lsol_imag.SetSubVector(int_idx,lsi_imag);

      GetElementVDofs(iel, vdofs);

      // complete all the dofs in the global vector
      sol_real.SetSubVector(vdofs,lsol_real);
      sol_imag.SetSubVector(vdofs,lsol_imag);
   }
}

ComplexBlockStaticCondensation::~ComplexBlockStaticCondensation()
{
   delete S_e_r; S_e_r = nullptr;
   delete S_e_i; S_e_i = nullptr;
   delete S_r; S_r = nullptr;
   delete S_i; S_i = nullptr;
   delete S; S=nullptr;
   delete y_r; y_r=nullptr;
   delete y_i; y_i=nullptr;
   delete y; y=nullptr;

   if (P) { delete P; } P=nullptr;
   if (R) { delete R; } R=nullptr;

#ifdef MFEM_USE_MPI
   if (parallel)
   {
      // The Complex Operator (S) is deleted above
      delete pS_e_r; pS_e_r=nullptr;
      delete pS_e_i; pS_e_i=nullptr;
      delete pS_r; pS_r=nullptr;
      delete pS_i; pS_i=nullptr;
      for (int i = 0; i<rblocks; i++)
      {
         delete ess_tdofs[i];
      }
      delete pP; pP = nullptr;
   }
#endif

   for (int i=0; i<lmat.Size(); i++)
   {
      delete lmat[i]; lmat[i] = nullptr;
      delete lvec[i]; lvec[i] = nullptr;
   }
}

}
