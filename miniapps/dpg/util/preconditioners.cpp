// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "preconditioners.hpp"

namespace mfem
{

#ifdef MFEM_USE_MPI

Solver * MakeFESpaceDefaultSolver(
   const ParFiniteElementSpace * pfespace, int print_level)
{
   FiniteElementCollection const &fec = *(pfespace->FEColl());
   const int vdim = pfespace->GetVDim();
   const int dim = pfespace->GetParMesh()->Dimension();
   Solver * prec = nullptr;
   if (dynamic_cast<const H1_FECollection*>(&fec) ||
       dynamic_cast<const L2_FECollection*>(&fec))
   {
      prec = new HypreBoomerAMG();
      dynamic_cast<HypreBoomerAMG*>(prec)->SetPrintLevel(print_level);
      if (vdim > 1)
      {
         dynamic_cast<HypreBoomerAMG*>(prec)->SetSystemsOptions(vdim);
      }
      return prec;
   }
   else if (dynamic_cast<const RT_FECollection*>(&fec) && dim == 3)
   {
      prec = new HypreADS(const_cast<ParFiniteElementSpace*>(pfespace));
      dynamic_cast<HypreADS*>(prec)->SetPrintLevel(print_level);
      return prec;
   }
   else if (dynamic_cast<const ND_FECollection*>(&fec) ||
            dynamic_cast<const RT_FECollection*>(&fec))
   {
      prec = new HypreAMS(const_cast<ParFiniteElementSpace*>(pfespace));
      dynamic_cast<HypreAMS*>(prec)->SetPrintLevel(print_level);
      return prec;
   }
   else
   {
      MFEM_ABORT("Unsupported FiniteElementCollection type");
   }
   return prec;
}


PRefinementHierarchy::PRefinementHierarchy(const Array<ParFiniteElementSpace*>
                                           &pfes_,
                                           const std::vector<Array<int>> & ess_bdr_marker_)
   : pfes(pfes_), ess_bdr_marker(ess_bdr_marker_), nblocks(pfes.Size())
{
   MFEM_VERIFY(nblocks > 0, "Empty pfes.");
   pmesh = pfes[0]->GetParMesh();
   MFEM_VERIFY(pmesh, "pfes[0] has null ParMesh.");
   MFEM_VERIFY(ess_bdr_marker.size() == static_cast<size_t>(nblocks),
               "ess_bdr_marker size must match nblocks.");
   int bdr_size = (pmesh->bdr_attributes.Size() > 0) ? pmesh->bdr_attributes.Max()
                  : 0;
   for (int i = 0; i<nblocks; i++)
   {
      MFEM_VERIFY(ess_bdr_marker[i].Size() == bdr_size,
                  "ess_bdr_marker[" << i << "] size must match max bdr_attribute in mesh.");
   }
}

const ParFiniteElementSpace* PRefinementHierarchy::GetParFESpace(int lev,
                                                                 int b) const
{
   if (lev == maxlevels - 1) { return pfes[b]; }
   return fes_owned[lev][b].get();
}

int PRefinementHierarchy::GetFESpaceMinimumOrder(const ParFiniteElementSpace
                                                 *pfespace)
const
{
   return (dynamic_cast<const L2_FECollection*>(pfespace->FEColl()) ||
           dynamic_cast<const RT_FECollection*>(pfespace->FEColl())) ? 0 : 1;
}

void PRefinementHierarchy::BuildSpaceHierarchy()
{
   orders.SetSize(nblocks);
   Array<int> levels(nblocks);
   for (int i = 0; i < nblocks; i++)
   {
      orders[i] = pfes[i]->FEColl()->GetConstructorOrder();
      levels[i] = orders[i] - GetFESpaceMinimumOrder(pfes[i]);
   }

   maxlevels = levels.Min() + 1;
   MFEM_VERIFY(maxlevels >= 1, "Invalid maxlevels computed.");

   fec_owned.resize(maxlevels-1);
   fes_owned.resize(maxlevels-1);
   T_level.resize(maxlevels-1);

   for (int lev = 0; lev < maxlevels-1; lev++)
   {
      fec_owned[lev].resize(nblocks);
      fes_owned[lev].resize(nblocks);
      T_level[lev].resize(nblocks);
   }

   // Build ParFES hierarchy for each block
   for (int b = 0; b < nblocks; b++)
   {
      const FiniteElementCollection *fec_ref = pfes[b]->FEColl();
      const int vdim = pfes[b]->GetVDim();
      const Ordering::Type ordering = pfes[b]->GetOrdering();

      for (int lev = 1; lev <= maxlevels - 1; lev++)
      {
         const int p = orders[b] - lev;

         auto &fec_ptr = fec_owned[maxlevels - lev - 1][b];
         auto &fes_ptr = fes_owned[maxlevels - lev - 1][b];

         fec_ptr.reset(fec_ref->Clone(p));
         fes_ptr = std::make_unique<ParFiniteElementSpace>(pmesh, fec_ptr.get(),
                                                           vdim, ordering);
      }
   }

   // build true dof lists for all levels
   ess_tdof_list.resize(maxlevels);
   Array<int> tdof_offsets(nblocks+1);
   for (int i = 0; i< maxlevels; i++)
   {
      tdof_offsets[0] = 0;
      for (int b = 0; b < nblocks; b++)
      {
         tdof_offsets[b+1] = GetParFESpace(i,b)->GetTrueVSize();
      }
      tdof_offsets.PartialSum();
      Array<int> tdof_list;
      Array<int> block_tdof_list;
      for (int b = 0; b < nblocks; b++)
      {
         block_tdof_list.SetSize(0);
         GetParFESpace(i,b)->GetEssentialTrueDofs(ess_bdr_marker[b], block_tdof_list);
         for (int j = 0; j < block_tdof_list.Size(); j++)
         {
            block_tdof_list[j] += tdof_offsets[b];
         }
         tdof_list.Append(block_tdof_list);
      }
      ess_tdof_list[i] = tdof_list;
   }
}

BlockOperator *PRefinementHierarchy::BuildProlongation(int lev)
{
   MFEM_VERIFY(lev >= 0 &&
               lev < maxlevels - 1, "Invalid level in BuildProlongation().");

   Array<int> coarse_offsets(nblocks + 1); coarse_offsets[0] = 0;
   Array<int> fine_offsets(nblocks + 1);   fine_offsets[0]   = 0;

   for (int b = 0; b < nblocks; b++)
   {
      coarse_offsets[b+1] = coarse_offsets[b] + GetParFESpace(lev,
                                                              b)->GetTrueVSize();
      fine_offsets[b+1]   = fine_offsets[b]   + GetParFESpace(lev+1,
                                                              b)->GetTrueVSize();
   }

   BlockOperator *Pblk = new BlockOperator(fine_offsets, coarse_offsets);
   Pblk->owns_blocks = 0;

   for (int b = 0; b < nblocks; b++)
   {
      T_level[lev][b] = std::make_unique<PRefinementTransferOperator>(
                           *GetParFESpace(lev, b), *GetParFESpace(lev+1, b), true);

      HypreParMatrix *P =
         dynamic_cast<HypreParMatrix*>(T_level[lev][b]->GetTrueTransferOperator());
      MFEM_VERIFY(P, "PRefinement transfer returned null.");
      Pblk->SetBlock(b, b, P);
   }
   return Pblk;
}


PRefinementMultigrid::PRefinementMultigrid(const Array<ParFiniteElementSpace*>
                                           &pfes_,
                                           const std::vector<Array<int>> & ess_bdr_marker_,
                                           const BlockOperator &Op_,
                                           bool mumps_coarse_solver)
   : Multigrid()
   , hierarchy(pfes_, ess_bdr_marker_)
   , Op(Op_)
{
#ifndef MFEM_USE_MUMPS
   if (mumps_coarse_solver)
   {
      MFEM_WARNING("MUMPS coarse solver requires MFEM built with MUMPS. Switching to default coarse solver.");
   }
   mumps_coarse_solver = false;
#endif

   hierarchy.BuildSpaceHierarchy();

   const int maxlevels = hierarchy.maxlevels;
   const int nblocks   = hierarchy.nblocks;

   operators.SetSize(maxlevels);
   ownedOperators.SetSize(maxlevels);
   smoothers.SetSize(maxlevels);
   ownedSmoothers.SetSize(maxlevels);

   operators[maxlevels-1] = const_cast<BlockOperator*>(&Op);
   ownedOperators[maxlevels-1] = false;

   const int nP = std::max(0, maxlevels - 1);
   prolongations.SetSize(nP);
   ownedProlongations.SetSize(nP);

   // Build prolongations and Galerkin operators
   for (int lev = nP - 1; lev >= 0; lev--)
   {
      BlockOperator *Pblk = hierarchy.BuildProlongation(lev);
      prolongations[lev] = new RectangularConstrainedOperator(Pblk,
                                                              hierarchy.ess_tdof_list[lev], hierarchy.ess_tdof_list[lev+1], true);
      ownedProlongations[lev] = true;

      BlockOperator *OpLevel = new BlockOperator(Pblk->ColOffsets());
      OpLevel->owns_blocks = 1;

      BlockOperator *OpFine = dynamic_cast<BlockOperator*>(operators[lev+1]);
      MFEM_VERIFY(OpFine, "Expected BlockOperator at fine level.");

      for (int i = 0; i < nblocks; i++)
      {
         HypreParMatrix *Pi = dynamic_cast<HypreParMatrix*>(&Pblk->GetBlock(i, i));
         MFEM_VERIFY(Pi, "Expected HypreParMatrix prolongation block.");
         HypreParMatrix *Pit = Pi->Transpose();

         for (int j = 0; j < nblocks; j++)
         {
            if (Op.IsZeroBlock(i, j)) { continue; }

            const HypreParMatrix *A_fine =
               dynamic_cast<const HypreParMatrix*>(&OpFine->GetBlock(i, j));
            MFEM_VERIFY(A_fine, "Expected HypreParMatrix block.");

            if (i == j)
            {
               OpLevel->SetBlock(i, i, RAP(A_fine, Pi));
            }
            else
            {
               HypreParMatrix *Pj = dynamic_cast<HypreParMatrix*>(&Pblk->GetBlock(j, j));
               MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");

               HypreParMatrix *APj  = ParMult(A_fine, Pj, true);
               HypreParMatrix *PtAP = ParMult(Pit, APj, true);
               delete APj;
               OpLevel->SetBlock(i, j, PtAP);
            }
         }
         delete Pit;
      }
      operators[lev] = OpLevel;
      ownedOperators[lev] = true;
   }

   // Build smoothers
   for (int lev = 0; lev < operators.Size(); lev++)
   {
      auto *cOp = dynamic_cast<BlockOperator*>(operators[lev]);
      MFEM_VERIFY(cOp, "Expected BlockOperator in operators[].");

      if (lev == 0) // coarse
      {
#ifdef MFEM_USE_MUMPS
         if (mumps_coarse_solver)
         {
            HypreParMatrix *Acoarse = cOp->GetMonolithicHypreParMatrix();
            auto *mumps_solver = new MUMPSSolver(MPI_COMM_WORLD);
            mumps_solver->SetPrintLevel(0);
            mumps_solver->SetOperator(*Acoarse);
            delete Acoarse;

            smoothers[lev] = mumps_solver;
            ownedSmoothers[lev] = true;
         }
         else
#endif
         {
            auto *bd = new BlockDiagonalPreconditioner(cOp->RowOffsets());
            bd->owns_blocks = 1;

            for (int b = 0; b < nblocks; b++)
            {
               const HypreParMatrix *Ab =
                  dynamic_cast<const HypreParMatrix*>(&cOp->GetBlock(b, b));
               MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

               auto solver = MakeFESpaceDefaultSolver(hierarchy.GetParFESpace(lev, b), 0);
               solver->SetOperator(*Ab);
               bd->SetDiagonalBlock(b, solver);
            }

            coarse_prec.reset(bd);

            auto *cg = new CGSolver(MPI_COMM_WORLD);
            cg->SetPrintLevel(-1);
            cg->SetRelTol(1e-3);
            cg->SetMaxIter(10);
            cg->SetOperator(*cOp);
            cg->SetPreconditioner(*coarse_prec);

            smoothers[lev] = cg;
            ownedSmoothers[lev] = true;
         }
      }
      else
      {
         auto *prec = new SymmetricBlockDiagonalPreconditioner(cOp->RowOffsets());
         prec->owns_blocks = 1;

         for (int b = 0; b < nblocks; b++)
         {
            const HypreParMatrix *Ab =
               dynamic_cast<const HypreParMatrix*>(&cOp->GetBlock(b, b));
            MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

            auto solver = MakeFESpaceDefaultSolver(hierarchy.GetParFESpace(lev, b), 0);
            solver->SetOperator(*Ab);
            prec->SetDiagonalBlock(b, solver);
         }
         smoothers[lev] = prec;
         ownedSmoothers[lev] = true;
      }
   }
}


ComplexPRefinementMultigrid::ComplexPRefinementMultigrid(
   const Array<ParFiniteElementSpace*> &pfes_,
   const std::vector<Array<int>> & ess_bdr_marker,
   const ComplexOperator &Op_, bool mumps_coarse_solver)
   : Multigrid(), Op(Op_)
{
#ifndef MFEM_USE_MUMPS
   if (mumps_coarse_solver)
   {
      MFEM_WARNING("MUMPS coarse solver requires MFEM built with MUMPS. Switching to default coarse solver.");
   }
   mumps_coarse_solver = false;
#endif

   const auto *Op_r = dynamic_cast<const BlockOperator*>(&Op.real());
   const auto *Op_i = dynamic_cast<const BlockOperator*>(&Op.imag());
   MFEM_VERIFY(Op_r, "Expected BlockOperator from ComplexOperator real part.");
   MFEM_VERIFY(Op_i, "Expected BlockOperator from ComplexOperator imag part.");

   const int nblocks = Op_r->NumRowBlocks();
   MFEM_VERIFY(nblocks == Op_i->NumRowBlocks(), "Real/imag block counts differ.");
   hierarchy = std::make_unique<PRefinementHierarchy>(pfes_, ess_bdr_marker);

   hierarchy->BuildSpaceHierarchy();

   const int maxlevels = hierarchy->maxlevels;

   operators.SetSize(maxlevels);
   ownedOperators.SetSize(maxlevels);
   smoothers.SetSize(maxlevels);
   ownedSmoothers.SetSize(maxlevels);

   operators[maxlevels-1] = const_cast<ComplexOperator*>(&Op);
   ownedOperators[maxlevels-1] = false;

   const int nP = std::max(0, maxlevels - 1);
   prolongations.SetSize(nP);
   ownedProlongations.SetSize(nP);

   for (int lev = nP - 1; lev >= 0; lev--)
   {
      BlockOperator *Pblk = hierarchy->BuildProlongation(lev);

      auto ConstrOp = new RectangularConstrainedOperator(Pblk,
                                                         hierarchy->ess_tdof_list[lev],
                                                         hierarchy->ess_tdof_list[lev+1], true);

      // prolongation as complex (real=Pblk, imag=nullptr)
      prolongations[lev] = new ComplexOperator(ConstrOp, nullptr, true, true);
      ownedProlongations[lev] = true;

      auto *OpLevel_r = new BlockOperator(Pblk->ColOffsets());
      auto *OpLevel_i = new BlockOperator(Pblk->ColOffsets());
      OpLevel_r->owns_blocks = 1;
      OpLevel_i->owns_blocks = 1;

      auto *cOp = dynamic_cast<ComplexOperator*>(operators[lev+1]);
      MFEM_VERIFY(cOp, "Expected ComplexOperator at fine level.");

      auto *cOp_r = dynamic_cast<BlockOperator*>(&cOp->real());
      auto *cOp_i = dynamic_cast<BlockOperator*>(&cOp->imag());
      MFEM_VERIFY(cOp_r, "Expected BlockOperator fine real part.");
      MFEM_VERIFY(cOp_i, "Expected BlockOperator fine imag part.");

      for (int i = 0; i < nblocks; i++)
      {
         HypreParMatrix *Pi = dynamic_cast<HypreParMatrix*>(&Pblk->GetBlock(i, i));
         MFEM_VERIFY(Pi, "Expected HypreParMatrix prolongation block.");
         HypreParMatrix *Pit = Pi->Transpose();

         for (int j = 0; j < nblocks; j++)
         {
            if (!Op_r->IsZeroBlock(i, j))
            {
               const HypreParMatrix *A_fine_r =
                  dynamic_cast<const HypreParMatrix*>(&cOp_r->GetBlock(i, j));
               MFEM_VERIFY(A_fine_r, "Expected HypreParMatrix block (real).");

               if (i == j)
               {
                  OpLevel_r->SetBlock(i, i, RAP(A_fine_r, Pi));
               }
               else
               {
                  HypreParMatrix *Pj = dynamic_cast<HypreParMatrix*>(&Pblk->GetBlock(j, j));
                  MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");

                  HypreParMatrix *APj  = ParMult(A_fine_r, Pj, true);
                  HypreParMatrix *PtAP = ParMult(Pit, APj, true);
                  delete APj;

                  OpLevel_r->SetBlock(i, j, PtAP);
               }
            }

            if (!Op_i->IsZeroBlock(i, j))
            {
               const HypreParMatrix *A_fine_i =
                  dynamic_cast<const HypreParMatrix*>(&cOp_i->GetBlock(i, j));
               MFEM_VERIFY(A_fine_i, "Expected HypreParMatrix block (imag).");

               if (i == j)
               {
                  OpLevel_i->SetBlock(i, i, RAP(A_fine_i, Pi));
               }
               else
               {
                  HypreParMatrix *Pj = dynamic_cast<HypreParMatrix*>(&Pblk->GetBlock(j, j));
                  MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");

                  HypreParMatrix *APj  = ParMult(A_fine_i, Pj, true);
                  HypreParMatrix *PtAP = ParMult(Pit, APj, true);
                  delete APj;

                  OpLevel_i->SetBlock(i, j, PtAP);
               }
            }
         }

         delete Pit;
      }

      auto *OpLevel_c = new ComplexOperator(OpLevel_r, OpLevel_i, true, true);
      operators[lev] = OpLevel_c;
      ownedOperators[lev] = true;
   }

   // smoothers
   for (int lev = 0; lev < operators.Size(); lev++)
   {
      auto *cOp = dynamic_cast<ComplexOperator*>(operators[lev]);
      MFEM_VERIFY(cOp, "Expected ComplexOperator in operators[].");

      auto *cOp_r = dynamic_cast<BlockOperator*>(&cOp->real());
      MFEM_VERIFY(cOp_r, "Expected BlockOperator real part in ComplexOperator.");

      if (lev == 0)
      {
#ifdef MFEM_USE_MUMPS
         if (mumps_coarse_solver)
         {
            ComplexHypreParMatrix *Ahc = cOp->AsComplexHypreParMatrix();
            HypreParMatrix *A = Ahc->GetSystemMatrix();
            delete Ahc;

            auto *mumps_solver = new MUMPSSolver(MPI_COMM_WORLD);
            mumps_solver->SetPrintLevel(0);
            mumps_solver->SetOperator(*A);
            delete A;

            smoothers[lev] = mumps_solver;
            ownedSmoothers[lev] = true;
         }
         else
#endif
         {
            auto *prec_r = new BlockDiagonalPreconditioner(cOp_r->RowOffsets());
            prec_r->owns_blocks = 1;

            for (int b = 0; b < nblocks; b++)
            {
               const HypreParMatrix *Ab =
                  dynamic_cast<const HypreParMatrix*>(&cOp_r->GetBlock(b, b));
               MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

               auto solver = MakeFESpaceDefaultSolver(hierarchy->GetParFESpace(lev, b), 0);
               solver->SetOperator(*Ab);
               prec_r->SetDiagonalBlock(b, solver);
            }

            coarse_prec.reset(new ComplexPreconditioner(prec_r, true));

            auto *cg = new CGSolver(MPI_COMM_WORLD);
            cg->SetPrintLevel(-1);
            cg->SetRelTol(1e-3);
            cg->SetMaxIter(10);
            cg->SetOperator(*cOp);
            cg->SetPreconditioner(*coarse_prec);

            smoothers[lev] = cg;
            ownedSmoothers[lev] = true;
         }
      }
      else
      {
         auto *prec_r = new SymmetricBlockDiagonalPreconditioner(cOp_r->RowOffsets());
         prec_r->owns_blocks = 1;

         for (int b = 0; b < nblocks; b++)
         {
            const HypreParMatrix *Ab =
               dynamic_cast<const HypreParMatrix*>(&cOp_r->GetBlock(b, b));
            MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

            auto solver = MakeFESpaceDefaultSolver(hierarchy->GetParFESpace(lev, b), 0);
            solver->SetOperator(*Ab);
            prec_r->SetDiagonalBlock(b, solver);
         }

         smoothers[lev] = new ComplexPreconditioner(prec_r, true);
         ownedSmoothers[lev] = true;
      }
   }
}

#endif

} // namespace mfem
