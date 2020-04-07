// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "staticcond.hpp"

namespace mfem
{

StaticCondensation::StaticCondensation(FiniteElementSpace *fespace)
   : fes(fespace)
{
   tr_fec = fespace->FEColl()->GetTraceCollection();
   int vdim = fes->GetVDim();
   int ordering = fes->GetOrdering();
#ifndef MFEM_USE_MPI
   tr_fes = new FiniteElementSpace(fes->GetMesh(), tr_fec, vdim, ordering);
#else
   pfes = dynamic_cast<ParFiniteElementSpace*>(fes);
   if (!pfes)
   {
      tr_fes = new FiniteElementSpace(fes->GetMesh(), tr_fec, vdim, ordering);
      tr_pfes = NULL;
   }
   else
   {
      tr_pfes = new ParFiniteElementSpace(pfes->GetParMesh(), tr_fec, vdim,
                                          ordering);
      tr_fes = tr_pfes;
   }
   pS.SetType(Operator::Hypre_ParCSR);
   pS_e.SetType(Operator::Hypre_ParCSR);
#endif
   S = S_e = NULL;
   symm = false;
   A_data.Reset();
   A_ipiv.Reset();

   Array<int> vdofs;
   const int NE = fes->GetNE();
   elem_pdof.MakeI(NE);
   for (int i = 0; i < NE; i++)
   {
      const int npd = vdim*fes->GetNumElementInteriorDofs(i);
      elem_pdof.AddColumnsInRow(i, npd);
   }
   elem_pdof.MakeJ();
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      const int nsd = vdofs.Size()/vdim;
      const int nspd = fes->GetNumElementInteriorDofs(i);
      const int *dofs = vdofs.GetData();
      for (int vd = 0; vd < vdim; vd++)
      {
#ifdef MFEM_DEBUG
         for (int j = 0; j < nspd; j++)
         {
            MFEM_ASSERT(dofs[nsd-nspd+j] >= 0, "");
         }
#endif
         elem_pdof.AddConnections(i, dofs+nsd-nspd, nspd);
         dofs += nsd;
      }
   }
   elem_pdof.ShiftUpI();
   // Set the number of private dofs.
   npdofs = elem_pdof.Size_of_connections();
   MFEM_ASSERT(fes->GetVSize() == tr_fes->GetVSize() + npdofs,
               "incompatible volume and trace FE spaces");
   // Initialize the map rdof_edof.
   rdof_edof.SetSize(tr_fes->GetVSize());
   Array<int> rvdofs;
   for (int i = 0; i < NE; i++)
   {
      fes->GetElementVDofs(i, vdofs);
      tr_fes->GetElementVDofs(i, rvdofs);
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
            rdof_edof[rvdof] = vdof;
         }
      }
   }
}

StaticCondensation::~StaticCondensation()
{
#ifdef MFEM_USE_MPI
   // pS, pS_e are automatically destroyed
#endif
   delete S_e;
   delete S;
   A_data.Delete();
   A_ipiv.Delete();
   delete tr_fes;
   delete tr_fec;
}

bool StaticCondensation::ReducesTrueVSize() const
{
   if (!Parallel())
   {
      return (tr_fes->GetTrueVSize() < fes->GetTrueVSize());
   }
   else
   {
#ifdef MFEM_USE_MPI
      return (tr_pfes->GlobalTrueVSize() < pfes->GlobalTrueVSize());
#else
      return false; // avoid compiler warning
#endif
   }
}

void StaticCondensation::Init(bool symmetric, bool block_diagonal)
{
   const int NE = fes->GetNE();
   // symm = symmetric; // TODO: handle the symmetric case
   A_offsets.SetSize(NE+1);
   A_ipiv_offsets.SetSize(NE+1);
   A_offsets[0] = A_ipiv_offsets[0] = 0;
   Array<int> rvdofs;
   for (int i = 0; i < NE; i++)
   {
      tr_fes->GetElementVDofs(i, rvdofs);
      const int ned = rvdofs.Size();
      const int npd = elem_pdof.RowSize(i);
      A_offsets[i+1] = A_offsets[i] + npd*(npd + (symm ? 1 : 2)*ned);
      A_ipiv_offsets[i+1] = A_ipiv_offsets[i] + npd;
   }
   A_data = Memory<double>(A_offsets[NE]);
   A_ipiv = Memory<int>(A_ipiv_offsets[NE]);
   const int nedofs = tr_fes->GetVSize();
   if (fes->GetVDim() == 1)
   {
      // The sparsity pattern of S is given by the map rdof->elem->rdof
      Table rdof_rdof;
      {
         Table elem_rdof, rdof_elem;
         elem_rdof.MakeI(NE);
         for (int i = 0; i < NE; i++)
         {
            tr_fes->GetElementVDofs(i, rvdofs);
            elem_rdof.AddColumnsInRow(i, rvdofs.Size());
         }
         elem_rdof.MakeJ();
         for (int i = 0; i < NE; i++)
         {
            tr_fes->GetElementVDofs(i, rvdofs);
            FiniteElementSpace::AdjustVDofs(rvdofs);
            elem_rdof.AddConnections(i, rvdofs.GetData(), rvdofs.Size());
         }
         elem_rdof.ShiftUpI();
         Transpose(elem_rdof, rdof_elem, nedofs);
         mfem::Mult(rdof_elem, elem_rdof, rdof_rdof);
      }
      S = new SparseMatrix(rdof_rdof.GetI(), rdof_rdof.GetJ(), NULL,
                           nedofs, nedofs, true, false, false);
      rdof_rdof.LoseData();
   }
   else
   {
      // For a block diagonal vector bilinear form, the sparsity of
      // rdof->elem->rdof is overkill, so we use dynamically allocated
      // sparsity pattern.
      S = new SparseMatrix(nedofs);
   }
}

void StaticCondensation::AssembleMatrix(int el, const DenseMatrix &elmat)
{
   Array<int> rvdofs;
   tr_fes->GetElementVDofs(el, rvdofs);
   const int vdim = fes->GetVDim();
   const int nvpd = elem_pdof.RowSize(el);
   const int nved = rvdofs.Size();
   DenseMatrix A_pp(A_data + A_offsets[el], nvpd, nvpd);
   DenseMatrix A_pe(A_pp.Data() + nvpd*nvpd, nvpd, nved);
   DenseMatrix A_ep;
   if (symm) { A_ep.SetSize(nved, nvpd); }
   else      { A_ep.UseExternalData(A_pe.Data() + nvpd*nved, nved, nvpd); }
   DenseMatrix A_ee(nved, nved);

   const int npd = nvpd/vdim;
   const int ned = nved/vdim;
   const int nd = npd + ned;
   // Copy the blocks from elmat to A_xx
   for (int i = 0; i < vdim; i++)
   {
      for (int j = 0; j < vdim; j++)
      {
         A_pp.CopyMN(elmat, npd, npd, i*nd+ned, j*nd+ned, i*npd, j*npd);
         A_pe.CopyMN(elmat, npd, ned, i*nd+ned, j*nd,     i*npd, j*ned);
         A_ep.CopyMN(elmat, ned, npd, i*nd,     j*nd+ned, i*ned, j*npd);
         A_ee.CopyMN(elmat, ned, ned, i*nd,     j*nd,     i*ned, j*ned);
      }
   }
   // Compute the Schur complement
   LUFactors lu(A_pp.Data(), A_ipiv + A_ipiv_offsets[el]);
   lu.Factor(nvpd);
   lu.BlockFactor(nvpd, nved, A_pe.Data(), A_ep.Data(), A_ee.Data());

   // Assemble the Schur complement
   const int skip_zeros = 0;
   S->AddSubMatrix(rvdofs, rvdofs, A_ee, skip_zeros);
}

void StaticCondensation::AssembleBdrMatrix(int el, const DenseMatrix &elmat)
{
   Array<int> rvdofs;
   tr_fes->GetBdrElementVDofs(el, rvdofs);
   const int skip_zeros = 0;
   S->AddSubMatrix(rvdofs, rvdofs, elmat, skip_zeros);
}

void StaticCondensation::Finalize()
{
   const int skip_zeros = 0;
   if (!Parallel())
   {
      S->Finalize(skip_zeros);
      if (S_e) { S_e->Finalize(skip_zeros); }
      const SparseMatrix *cP = tr_fes->GetConformingProlongation();
      if (cP)
      {
         if (S->Height() != cP->Width())
         {
            SparseMatrix *cS = mfem::RAP(*cP, *S, *cP);
            delete S;
            S = cS;
         }
         if (S_e && S_e->Height() != cP->Width())
         {
            SparseMatrix *cS_e = mfem::RAP(*cP, *S_e, *cP);
            delete S_e;
            S = cS_e;
         }
      }
   }
   else // parallel
   {
#ifdef MFEM_USE_MPI
      if (!S) { return; } // already finalized
      S->Finalize(skip_zeros);
      if (S_e) { S_e->Finalize(skip_zeros); }
      OperatorHandle dS(pS.Type()), pP(pS.Type());
      dS.MakeSquareBlockDiag(tr_pfes->GetComm(), tr_pfes->GlobalVSize(),
                             tr_pfes->GetDofOffsets(), S);
      // TODO - construct Dof_TrueDof_Matrix directly in the pS format
      pP.ConvertFrom(tr_pfes->Dof_TrueDof_Matrix());
      pS.MakePtAP(dS, pP);
      dS.Clear();
      delete S;
      S = NULL;
      if (S_e)
      {
         OperatorHandle dS_e(pS_e.Type());
         dS_e.MakeSquareBlockDiag(tr_pfes->GetComm(), tr_pfes->GlobalVSize(),
                                  tr_pfes->GetDofOffsets(), S_e);
         pS_e.MakePtAP(dS_e, pP);
         dS_e.Clear();
         delete S_e;
         S_e = NULL;
      }
#endif
   }
}

void StaticCondensation::EliminateReducedTrueDofs(
   const Array<int> &ess_rtdof_list, Matrix::DiagonalPolicy dpolicy)
{
   if (!Parallel() || S) // not parallel or not finalized
   {
      if (S_e == NULL)
      {
         S_e = new SparseMatrix(S->Height());
      }
      for (int i = 0; i < ess_rtdof_list.Size(); i++)
      {
         S->EliminateRowCol(ess_rtdof_list[i], *S_e, dpolicy);
      }
   }
   else // parallel and finalized
   {
#ifdef MFEM_USE_MPI
      MFEM_ASSERT(pS_e.Ptr() == NULL, "essential b.c. already eliminated");
      pS_e.EliminateRowsCols(pS, ess_rtdof_list);
#endif
   }
}

void StaticCondensation::ReduceRHS(const Vector &b, Vector &sc_b) const
{
   // sc_b = b_e - A_ep A_pp_inv b_p

   MFEM_ASSERT(b.Size() == fes->GetVSize(), "'b' has incorrect size");

   const int NE = fes->GetNE();
   const int nedofs = tr_fes->GetVSize();
   const SparseMatrix *tr_cP = NULL;
   Vector b_r;
   if (!Parallel() && !(tr_cP = tr_fes->GetConformingProlongation()))
   {
      sc_b.SetSize(nedofs);
      b_r.SetDataAndSize(sc_b.GetData(), sc_b.Size());
   }
   else
   {
      b_r.SetSize(nedofs);
   }
   for (int i = 0; i < nedofs; i++)
   {
      b_r(i) = b(rdof_edof[i]);
   }

   DenseMatrix U_pe, L_ep;
   Vector b_p, b_ep;
   Array<int> rvdofs;
   for (int i = 0; i < NE; i++)
   {
      tr_fes->GetElementVDofs(i, rvdofs);
      const int ned = rvdofs.Size();
      const int *rd = rvdofs.GetData();
      const int npd = elem_pdof.RowSize(i);
      const int *pd = elem_pdof.GetRow(i);
      b_p.SetSize(npd);
      b_ep.SetSize(ned);
      for (int j = 0; j < npd; j++)
      {
         b_p(j) = b(pd[j]);
      }

      LUFactors lu(const_cast<double*>((const double*)A_data) + A_offsets[i],
                   const_cast<int*>((const int*)A_ipiv) + A_ipiv_offsets[i]);
      lu.LSolve(npd, 1, b_p);

      if (symm)
      {
         // TODO: handle the symmetric case correctly.
         U_pe.UseExternalData(lu.data + npd*npd, npd, ned);
         U_pe.MultTranspose(b_p, b_ep);
      }
      else
      {
         L_ep.UseExternalData(lu.data + npd*(npd+ned), ned, npd);
         L_ep.Mult(b_p, b_ep);
      }
      for (int j = 0; j < ned; j++)
      {
         if (rd[j] >= 0) { b_r(rd[j]) -= b_ep(j); }
         else            { b_r(-1-rd[j]) += b_ep(j); }
      }
   }
   if (!Parallel())
   {
      if (tr_cP)
      {
         sc_b.SetSize(tr_cP->Width());
         tr_cP->MultTranspose(b_r, sc_b);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      const Operator *tr_P = tr_pfes->GetProlongationMatrix();
      sc_b.SetSize(tr_P->Width());
      tr_P->MultTranspose(b_r, sc_b);
#endif
   }
}

void StaticCondensation::ReduceSolution(const Vector &sol, Vector &sc_sol) const
{
   MFEM_ASSERT(sol.Size() == fes->GetVSize(), "'sol' has incorrect size");

   const int nedofs = tr_fes->GetVSize();
   const SparseMatrix *tr_R = tr_fes->GetRestrictionMatrix();
   Vector sol_r;
   if (!tr_R)
   {
      sc_sol.SetSize(nedofs);
      sol_r.SetDataAndSize(sc_sol.GetData(), sc_sol.Size());
   }
   else
   {
      sol_r.SetSize(nedofs);
   }
   for (int i = 0; i < nedofs; i++)
   {
      sol_r(i) = sol(rdof_edof[i]);
   }
   if (tr_R)
   {
      sc_sol.SetSize(tr_R->Height());
      tr_R->Mult(sol_r, sc_sol);
   }
}

void StaticCondensation::ReduceSystem(Vector &x, Vector &b, Vector &X,
                                      Vector &B, int copy_interior) const
{
   ReduceRHS(b, B);
   ReduceSolution(x, X);
   if (!Parallel())
   {
      S_e->AddMult(X, B, -1.);
      S->PartMult(ess_rtdof_list, X, B);
   }
   else
   {
#ifdef MFEM_USE_MPI
      MFEM_ASSERT(pS.Type() == pS_e.Type(), "type id mismatch");
      pS.EliminateBC(pS_e, ess_rtdof_list, X, B);
#endif
   }
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_rtdof_list, 0.0);
   }
}

void StaticCondensation::ConvertMarkerToReducedTrueDofs(
   const Array<int> &ess_tdof_marker, Array<int> &ess_rtdof_marker) const
{
   const int nedofs = tr_fes->GetVSize();
   const SparseMatrix *R = fes->GetRestrictionMatrix();
   Array<int> ess_dof_marker;
   if (!R)
   {
      ess_dof_marker.MakeRef(ess_tdof_marker);
   }
   else
   {
      ess_dof_marker.SetSize(fes->GetVSize());
      R->BooleanMultTranspose(ess_tdof_marker, ess_dof_marker);
   }
   const SparseMatrix *tr_R = tr_fes->GetRestrictionMatrix();
   Array<int> ess_rdof_marker;
   if (!tr_R)
   {
      ess_rtdof_marker.SetSize(nedofs);
      ess_rdof_marker.MakeRef(ess_rtdof_marker);
   }
   else
   {
      ess_rdof_marker.SetSize(nedofs);
   }
   for (int i = 0; i < nedofs; i++)
   {
      ess_rdof_marker[i] = ess_dof_marker[rdof_edof[i]];
   }
   if (tr_R)
   {
      ess_rtdof_marker.SetSize(tr_R->Height());
      tr_R->BooleanMult(ess_rdof_marker, ess_rtdof_marker);
   }
}

void StaticCondensation::ComputeSolution(
   const Vector &b, const Vector &sc_sol, Vector &sol) const
{
   // sol_e = sc_sol
   // sol_p = A_pp_inv (b_p - A_pe sc_sol)

   MFEM_ASSERT(b.Size() == fes->GetVSize(), "'b' has incorrect size");

   const int nedofs = tr_fes->GetVSize();
   Vector sol_r;
   if (!Parallel())
   {
      const SparseMatrix *tr_cP = tr_fes->GetConformingProlongation();
      if (!tr_cP)
      {
         sol_r.SetDataAndSize(sc_sol.GetData(), sc_sol.Size());
      }
      else
      {
         sol_r.SetSize(nedofs);
         tr_cP->Mult(sc_sol, sol_r);
      }
   }
   else
   {
#ifdef MFEM_USE_MPI
      sol_r.SetSize(nedofs);
      tr_pfes->GetProlongationMatrix()->Mult(sc_sol, sol_r);
#endif
   }
   sol.SetSize(nedofs+npdofs);
   for (int i = 0; i < nedofs; i++)
   {
      sol(rdof_edof[i]) = sol_r(i);
   }
   const int NE = fes->GetNE();
   Vector b_p, s_e;
   Array<int> rvdofs;
   for (int i = 0; i < NE; i++)
   {
      tr_fes->GetElementVDofs(i, rvdofs);
      const int ned = rvdofs.Size();
      const int npd = elem_pdof.RowSize(i);
      const int *pd = elem_pdof.GetRow(i);
      b_p.SetSize(npd);

      for (int j = 0; j < npd; j++)
      {
         b_p(j) = b(pd[j]);
      }
      sol_r.GetSubVector(rvdofs, s_e);

      LUFactors lu(const_cast<double*>((const double*)A_data) + A_offsets[i],
                   const_cast<int*>((const int*)A_ipiv) + A_ipiv_offsets[i]);
      lu.LSolve(npd, 1, b_p);
      lu.BlockBackSolve(npd, ned, 1, lu.data + npd*npd, s_e, b_p);

      for (int j = 0; j < npd; j++)
      {
         sol(pd[j]) = b_p(j);
      }
   }
}

}
