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

#ifdef MFEM_USE_MPI

PRefinementMultigrid::PRefinementMultigrid(const Array<ParFiniteElementSpace *>
                                           & pfes_, const BlockOperator & Op_)
   : Multigrid(), pfes(pfes_), npfes(pfes.Size()), Op(Op_)
{
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
      SymmetricBlockDiagonalPreconditioner *prec =
         new SymmetricBlockDiagonalPreconditioner(dynamic_cast<BlockOperator*>
                                                  (operators[i])->RowOffsets());
      prec->owns_blocks = 1;
      for (int b = 0; b < nblocks; b++)
      {
         const HypreParMatrix * Ab = dynamic_cast<const HypreParMatrix *>
                                     (&dynamic_cast<BlockOperator*>(operators[i])->GetBlock(b, b));
         MFEM_VERIFY(Ab, "Expected HypreParMatrix block.");

         auto solver = MakeFESpaceDefaultSolver(GetParFESpace(i,b), 0);
         solver->SetOperator(*Ab);
         prec->SetDiagonalBlock(b, solver);
      }
      smoothers[i] = prec;
      ownedSmoothers[i] = true;
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
}


#endif

} // namespace mfem

