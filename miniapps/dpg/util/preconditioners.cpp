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

PRefinementMultigrid::PRefinementMultigrid(const Array<ParFiniteElementSpace *>
                                           & pfes_, const BlockOperator & Op_,
                                           bool mumps_coarse_solver)
   : Multigrid(), pfes(pfes_), npfes(pfes.Size()), Op(Op_)
{
#ifndef MFEM_USE_MUMPS
   if (mumps_coarse_solver)
   {
      MFEM_WARNING("MUMPS coarse solver requires MFEM built with MPI. Switching to default coarse solver.");
   }
   mumps_coarse_solver = false;
#endif
   nblocks = Op.NumRowBlocks();
   MFEM_VERIFY(npfes == nblocks, "pfes size must match Op.NumRowBlocks()");
   orders.SetSize(npfes);
   pmesh = pfes[0]->GetParMesh();
   Array<int> levels(npfes);
   for (int i = 0; i < npfes; i++)
   {
      orders[i] = pfes[i]->FEColl()->GetConstructorOrder();
      levels[i] = orders[i] - GetMinimumOrder(pfes[i]);
   }
   maxlevels = levels.Min()+1;
   // initialize the space hierarchy
   fec_owned.resize(maxlevels-1);
   fes_owned.resize(maxlevels-1);
   T_level.resize(maxlevels-1);
   for (int lev = 0; lev < maxlevels-1; lev++)
   {
      fec_owned[lev].resize(npfes);
      fes_owned[lev].resize(npfes);
      for (int b = 0; b < npfes; b++)
      {
         fec_owned[lev][b] = nullptr;
         fes_owned[lev][b] = nullptr;
      }
   }
   operators.SetSize(maxlevels);
   ownedOperators.SetSize(maxlevels);
   smoothers.SetSize(maxlevels);
   ownedSmoothers.SetSize(maxlevels);

   operators[maxlevels-1] = const_cast<BlockOperator*>(&Op);
   ownedOperators[maxlevels-1] = false;
   // build ParFES hierarchy for each block
   for (int b = 0; b < npfes; b++)
   {
      const FiniteElementCollection *fec_ref = pfes[b]->FEColl();
      const int vdim = pfes[b]->GetVDim();
      const Ordering::Type ordering = pfes[b]->GetOrdering();

      for (int lev = 1; lev <= maxlevels-1; lev++)
      {
         const int p = orders[b] - lev;
         fec_owned[maxlevels-lev-1][b] = std::unique_ptr<FiniteElementCollection>
                                         (fec_ref->Clone(p));
         fes_owned[maxlevels-lev-1][b] = std::make_unique<ParFiniteElementSpace>(pmesh,
                                                                                 fec_owned[maxlevels-lev-1][b].get(),
                                                                                 vdim, ordering);
      }
   }

   // 2) build prolongations: block-diagonal, one BlockOperator per level
   const int nP = std::max(0, maxlevels - 1);
   prolongations.SetSize(nP);
   ownedProlongations.SetSize(nP);

   for (int lev = nP-1; lev >=0; lev--)
   {
      T_level[lev].resize(nblocks);

      Array<int> coarse_offsets(nblocks + 1); coarse_offsets[0] = 0;
      Array<int> fine_offsets(nblocks + 1);   fine_offsets[0]   = 0;

      for (int b = 0; b < nblocks; b++)
      {
         coarse_offsets[b+1] = coarse_offsets[b] + GetParFESpace(lev,b)->GetTrueVSize();
         fine_offsets[b+1]   = fine_offsets[b]   + GetParFESpace(lev+1,
                                                                 b)->GetTrueVSize();
      }

      BlockOperator *Pblk = new BlockOperator(fine_offsets, coarse_offsets);
      Pblk->owns_blocks = 0;

      for (int b = 0; b < nblocks; b++)
      {
         T_level[lev][b] = new PRefinementTransferOperator(*GetParFESpace(lev,b),
                                                           *GetParFESpace(lev+1,b), true);

         HypreParMatrix *P = dynamic_cast<HypreParMatrix *>
                             (T_level[lev][b]->GetTrueTransferOperator());
         MFEM_VERIFY(P, "Prefinement transfer returned null.");
         Pblk->SetBlock(b, b, P);
      }
      prolongations[lev] = Pblk;
      ownedProlongations[lev] = true;
      // Build BlockOperator at each level
      BlockOperator * OpLevel = new BlockOperator(Pblk->ColOffsets());
      OpLevel->owns_blocks = 1;
      for (int i = 0; i < nblocks; i++)
      {
         BlockOperator * blkp = dynamic_cast<BlockOperator*>(prolongations[lev]);
         HypreParMatrix * Pi = dynamic_cast<HypreParMatrix *>(&blkp->GetBlock(i, i));
         MFEM_VERIFY(Pi, "Expected HypreParMatrix prolongation block.");
         HypreParMatrix * Pit = Pi->Transpose();
         for (int j = 0; j < nblocks; j++)
         {
            if (Op.IsZeroBlock(i, j)) { continue; }
            BlockOperator * blkop = dynamic_cast<BlockOperator*>(operators[lev+1]);
            const HypreParMatrix * A_fine = dynamic_cast<const HypreParMatrix *>
                                            (&blkop->GetBlock(i, j));
            MFEM_VERIFY(A_fine, "Expected HypreParMatrix block.");

            if (i == j)
            {
               HypreParMatrix * tmp = RAP(A_fine, Pi);
               OpLevel->SetBlock(i, i, tmp);
            }
            else
            {
               HypreParMatrix * Pj = dynamic_cast<HypreParMatrix *>(&blkp->GetBlock(j, j));
               MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");
               HypreParMatrix * APj = ParMult(A_fine, Pj,true);
               HypreParMatrix * PtAP = ParMult(Pit, APj,true);
               delete APj;
               OpLevel->SetBlock(i, j, PtAP);
            }
         }
         delete Pit;
      }
      operators[lev] = OpLevel;
      ownedOperators[lev] = true;
   }

   for (int i = 0; i < operators.Size(); i++)
   {
      auto cOp = dynamic_cast<BlockOperator*>(operators[i]);
      if (i == 0) // coarse level
      {
#ifdef MFEM_USE_MUMPS
         if (mumps_coarse_solver)
         {
            HypreParMatrix * Acoarse = cOp->GetMonolithicHypreParMatrix();
            auto mumps_solver = new MUMPSSolver(MPI_COMM_WORLD);
            mumps_solver->SetPrintLevel(0);
            mumps_solver->SetOperator(*Acoarse);
            delete Acoarse;
            smoothers[i] = mumps_solver;
            ownedSmoothers[i] = true;
         }
         else
#endif
         {
            coarse_prec = new BlockDiagonalPreconditioner(cOp->RowOffsets());
            BlockDiagonalPreconditioner * blog_diag =
               dynamic_cast<BlockDiagonalPreconditioner*>(coarse_prec);
            blog_diag->owns_blocks = 1;
            for (int b = 0; b < nblocks; b++)
            {
               const HypreParMatrix * Ab = dynamic_cast<const HypreParMatrix *>
                                           (&cOp->GetBlock(b, b));
               MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");
               auto solver = MakeFESpaceDefaultSolver(GetParFESpace(i,b), 0);
               solver->SetOperator(*Ab);
               blog_diag->SetDiagonalBlock(b, solver);
            }

            CGSolver * cg_solver = new CGSolver(MPI_COMM_WORLD);
            cg_solver->SetPrintLevel(-1);
            cg_solver->SetRelTol(1e-3);
            cg_solver->SetMaxIter(10);
            cg_solver->SetOperator(*cOp);
            cg_solver->SetPreconditioner(*blog_diag);
            smoothers[i] = cg_solver;
            ownedSmoothers[i] = true;
         }
      }
      else
      {
         SymmetricBlockDiagonalPreconditioner *prec =
            new SymmetricBlockDiagonalPreconditioner(cOp->RowOffsets());
         prec->owns_blocks = 1;
         for (int b = 0; b < nblocks; b++)
         {
            const HypreParMatrix * Ab = dynamic_cast<const HypreParMatrix *>
                                        (&cOp->GetBlock(b, b));
            MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

            auto solver = MakeFESpaceDefaultSolver(GetParFESpace(i,b), 0);
            solver->SetOperator(*Ab);
            prec->SetDiagonalBlock(b, solver);
         }
         smoothers[i] = prec;
         ownedSmoothers[i] = true;
      }
   }
}

PRefinementMultigrid::~PRefinementMultigrid()
{
   for (auto &lvl : T_level)
   {
      const int nb = static_cast<int>(lvl.size());
      for (int b = 0; b < nb; b++)
      {
         delete lvl[b];
      }
   }
   if (coarse_prec)
   {
      delete coarse_prec;
   }
}

ComplexPRefinementMultigrid::ComplexPRefinementMultigrid(
   const Array<ParFiniteElementSpace *>
   & pfes_, const ComplexOperator & Op_, bool mumps_coarse_solver)
   : Multigrid(), pfes(pfes_), npfes(pfes.Size()), Op(Op_)
{
#ifndef MFEM_USE_MUMPS
   if (mumps_coarse_solver)
   {
      MFEM_WARNING("MUMPS coarse solver requires MFEM built with MPI. Switching to default coarse solver.");
   }
   mumps_coarse_solver = false;
#endif

   const BlockOperator * Op_r = dynamic_cast<const BlockOperator *>(&Op.real());
   const BlockOperator * Op_i = dynamic_cast<const BlockOperator *>(&Op.imag());
   MFEM_VERIFY(Op_r, "Expected BlockOperator from ComplexOperator real part.");
   MFEM_VERIFY(Op_i,
               "Expected BlockOperator from ComplexOperator imaginary part.");
   nblocks = Op_r->NumRowBlocks();
   MFEM_VERIFY(npfes == nblocks, "pfes size must match Op.NumRowBlocks()");
   orders.SetSize(npfes);
   pmesh = pfes[0]->GetParMesh();
   Array<int> levels(npfes);
   for (int i = 0; i < npfes; i++)
   {
      orders[i] = pfes[i]->FEColl()->GetConstructorOrder();
      levels[i] = orders[i] - GetMinimumOrder(pfes[i]);
   }
   maxlevels = levels.Min()+1;
   // initialize the space hierarchy
   fec_owned.resize(maxlevels-1);
   fes_owned.resize(maxlevels-1);
   T_level.resize(maxlevels-1);
   for (int lev = 0; lev < maxlevels-1; lev++)
   {
      fec_owned[lev].resize(npfes);
      fes_owned[lev].resize(npfes);
      for (int b = 0; b < npfes; b++)
      {
         fec_owned[lev][b] = nullptr;
         fes_owned[lev][b] = nullptr;
      }
   }
   operators.SetSize(maxlevels);
   ownedOperators.SetSize(maxlevels);
   smoothers.SetSize(maxlevels);
   ownedSmoothers.SetSize(maxlevels);

   operators[maxlevels-1] = const_cast<ComplexOperator*>(&Op);
   ownedOperators[maxlevels-1] = false;

   // build ParFES hierarchy for each block
   for (int b = 0; b < npfes; b++)
   {
      const FiniteElementCollection *fec_ref = pfes[b]->FEColl();
      const int vdim = pfes[b]->GetVDim();
      const Ordering::Type ordering = pfes[b]->GetOrdering();

      for (int lev = 1; lev <= maxlevels-1; lev++)
      {
         const int p = orders[b] - lev;
         fec_owned[maxlevels-lev-1][b] = std::unique_ptr<FiniteElementCollection>
                                         (fec_ref->Clone(p));
         fes_owned[maxlevels-lev-1][b] = std::make_unique<ParFiniteElementSpace>(pmesh,
                                                                                 fec_owned[maxlevels-lev-1][b].get(),
                                                                                 vdim, ordering);
      }
   }

   // 2) build prolongations: block-diagonal, one Complex BlockOperator per level
   const int nP = std::max(0, maxlevels - 1);
   prolongations.SetSize(nP);
   ownedProlongations.SetSize(nP);

   for (int lev = nP-1; lev >=0; lev--)
   {
      T_level[lev].resize(nblocks);

      Array<int> coarse_offsets(nblocks + 1); coarse_offsets[0] = 0;
      Array<int> fine_offsets(nblocks + 1);   fine_offsets[0]   = 0;

      for (int b = 0; b < nblocks; b++)
      {
         coarse_offsets[b+1] = coarse_offsets[b] + GetParFESpace(lev,b)->GetTrueVSize();
         fine_offsets[b+1]   = fine_offsets[b]   + GetParFESpace(lev+1,
                                                                 b)->GetTrueVSize();
      }

      BlockOperator *Pblk = new BlockOperator(fine_offsets, coarse_offsets);
      Pblk->owns_blocks = 0;

      for (int b = 0; b < nblocks; b++)
      {
         T_level[lev][b] = new PRefinementTransferOperator(*GetParFESpace(lev,b),
                                                           *GetParFESpace(lev+1,b), true);
         HypreParMatrix *P = dynamic_cast<HypreParMatrix *>
                             (T_level[lev][b]->GetTrueTransferOperator());
         MFEM_VERIFY(P, "Prefinement transfer returned null.");
         Pblk->SetBlock(b, b, P);
      }

      ComplexOperator * CPblk = new ComplexOperator(Pblk, nullptr, true, true);
      prolongations[lev] = CPblk;
      ownedProlongations[lev] = true;
      // Build ComplexOperator at each level
      BlockOperator * OpLevel_r = new BlockOperator(Pblk->ColOffsets());
      BlockOperator * OpLevel_i = new BlockOperator(Pblk->ColOffsets());
      OpLevel_r->owns_blocks = 1;
      OpLevel_i->owns_blocks = 1;

      for (int i = 0; i < nblocks; i++)
      {
         auto cP = dynamic_cast<ComplexOperator*>(prolongations[lev]);
         BlockOperator * blkp = dynamic_cast<BlockOperator*>(&cP->real());
         HypreParMatrix * Pi = dynamic_cast<HypreParMatrix *>(&blkp->GetBlock(i, i));
         MFEM_VERIFY(Pi, "Expected HypreParMatrix prolongation block.");
         HypreParMatrix * Pit = Pi->Transpose();
         for (int j = 0; j < nblocks; j++)
         {
            if (!Op_r->IsZeroBlock(i, j))
            {
               auto cOp = dynamic_cast<ComplexOperator*>(operators[lev+1]);
               BlockOperator * blkop_r = dynamic_cast<BlockOperator*>(&cOp->real());
               const HypreParMatrix * A_fine_r = dynamic_cast<const HypreParMatrix *>
                                                 (&blkop_r->GetBlock(i, j));
               MFEM_VERIFY(A_fine_r, "Expected HypreParMatrix block.");
               if (i == j)
               {
                  HypreParMatrix * tmp = RAP(A_fine_r, Pi);
                  OpLevel_r->SetBlock(i, i, tmp);
               }
               else
               {
                  HypreParMatrix * Pj = dynamic_cast<HypreParMatrix *>(&blkp->GetBlock(j, j));
                  MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");
                  HypreParMatrix * APj = ParMult(A_fine_r, Pj,true);
                  HypreParMatrix * PtAP = ParMult(Pit, APj,true);
                  delete APj;
                  OpLevel_r->SetBlock(i, j, PtAP);
               }

            }
            if (!Op_i->IsZeroBlock(i, j))
            {
               auto cOp = dynamic_cast<ComplexOperator*>(operators[lev+1]);
               BlockOperator * blkop_i = dynamic_cast<BlockOperator*>(&cOp->imag());
               const HypreParMatrix * A_fine_i = dynamic_cast<const HypreParMatrix *>
                                                 (&blkop_i->GetBlock(i, j));
               MFEM_VERIFY(A_fine_i, "Expected HypreParMatrix block.");
               if (i == j)
               {
                  HypreParMatrix * tmp = RAP(A_fine_i, Pi);
                  OpLevel_i->SetBlock(i, i, tmp);
               }
               else
               {
                  HypreParMatrix * Pj = dynamic_cast<HypreParMatrix *>(&blkp->GetBlock(j, j));
                  MFEM_VERIFY(Pj, "Expected HypreParMatrix prolongation block.");
                  HypreParMatrix * APj = ParMult(A_fine_i, Pj,true);
                  HypreParMatrix * PtAP = ParMult(Pit, APj,true);
                  delete APj;
                  OpLevel_i->SetBlock(i, j, PtAP);
               }
            }
         }
         delete Pit;
      }

      ComplexOperator * OpLevel_c = new ComplexOperator(OpLevel_r, OpLevel_i,
                                                        true, true);
      operators[lev] = OpLevel_c;
      ownedOperators[lev] = true;
   }

   for (int i = 0; i < operators.Size(); i++)
   {
      auto cOp = dynamic_cast<ComplexOperator*>(operators[i]);
      if (i == 0) // coarse level
      {
#ifdef MFEM_USE_MUMPS
         if (mumps_coarse_solver)
         {
            ComplexHypreParMatrix * Ahc = cOp->AsComplexHypreParMatrix();
            HypreParMatrix * A = Ahc->GetSystemMatrix();
            delete Ahc;
            auto mumps_solver = new MUMPSSolver(MPI_COMM_WORLD);
            mumps_solver->SetOperator(*A);
            mumps_solver->SetPrintLevel(0);
            delete A;
            smoothers[i] = mumps_solver;
            ownedSmoothers[i] = true;
         }
         else
#endif
         {
            BlockDiagonalPreconditioner * prec =
               new BlockDiagonalPreconditioner(dynamic_cast<BlockOperator*>
                                               (&cOp->real())->RowOffsets());
            prec->owns_blocks = 1;
            for (int b = 0; b < nblocks; b++)
            {
               const HypreParMatrix * Ab = dynamic_cast<const HypreParMatrix *>
                                           (&dynamic_cast<BlockOperator*>(&cOp->real())->GetBlock(b, b));
               MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");
               auto solver = MakeFESpaceDefaultSolver(GetParFESpace(i,b), 0);
               solver->SetOperator(*Ab);
               prec->SetDiagonalBlock(b, solver);
            }
            coarse_prec = new ComplexPreconditioner(prec, true);
            CGSolver * cg_solver = new CGSolver(MPI_COMM_WORLD);
            cg_solver->SetPrintLevel(-1);
            cg_solver->SetRelTol(1e-3);
            cg_solver->SetMaxIter(10);
            cg_solver->SetOperator(*cOp);
            cg_solver->SetPreconditioner(*coarse_prec);
            smoothers[i] = cg_solver;
            ownedSmoothers[i] = true;
         }
      }
      else
      {
         SymmetricBlockDiagonalPreconditioner *prec =
            new SymmetricBlockDiagonalPreconditioner(dynamic_cast<BlockOperator*>
                                                     (&cOp->real())->RowOffsets());
         prec->owns_blocks = 1;
         for (int b = 0; b < nblocks; b++)
         {
            const HypreParMatrix * Ab = dynamic_cast<const HypreParMatrix *>
                                        (&dynamic_cast<BlockOperator*>(&cOp->real())->GetBlock(b, b));
            MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

            auto solver = MakeFESpaceDefaultSolver(GetParFESpace(i,b), 0);
            solver->SetOperator(*Ab);
            prec->SetDiagonalBlock(b, solver);
         }
         smoothers[i] = new ComplexPreconditioner(prec, true);
         ownedSmoothers[i] = true;
      }
   }
}

ComplexPRefinementMultigrid::~ComplexPRefinementMultigrid()
{
   for (auto &lvl : T_level)
   {
      const int nb = static_cast<int>(lvl.size());
      for (int b = 0; b < nb; b++)
      {
         delete lvl[b];
      }
   }
   if (coarse_prec)
   {
      delete coarse_prec;
   }
}


#endif

} // namespace mfem

