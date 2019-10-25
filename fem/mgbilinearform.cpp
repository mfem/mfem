// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mgbilinearform.hpp"
#include "transfer.hpp"

namespace mfem
{

MultigridBilinearForm::MultigridBilinearForm(SpaceHierarchy& spaceHierarchy,
                                             BilinearForm& bf,
                                             Array<int>& ess_bdr)
    : MultigridOperator()
{
   MFEM_VERIFY(bf.GetAssemblyLevel() == AssemblyLevel::PARTIAL,
               "Assembly level must be PARTIAL");
   SetupPA(spaceHierarchy, bf, ess_bdr);
}

MultigridBilinearForm::MultigridBilinearForm(SpaceHierarchy& spaceHierarchy,
                                             SparseMatrix& opr, Array<int>& ess_bdr)
    : MultigridOperator()
{
   SetupFull(spaceHierarchy, opr, ess_bdr);
}

MultigridBilinearForm::~MultigridBilinearForm()
{
   for (int i = 0; i < bfs.Size(); ++i)
   {
      delete bfs[i];
   }

   for (int i = 0; i < essentialTrueDofs.Size(); ++i)
   {
      delete essentialTrueDofs[i];
   }
}

void MultigridBilinearForm::SetupPA(SpaceHierarchy& spaceHierarchy,
                                    BilinearForm& bf, Array<int>& ess_bdr)
{
   BilinearForm* form = new BilinearForm(&spaceHierarchy.GetFESpaceAtLevel(0));
   // TODO: Copy all integrators
   Array<BilinearFormIntegrator*>& dbfi = *bf.GetDBFI();
   for (int i = 0; i < dbfi.Size(); ++i)
   {
      form->AddDomainIntegrator((*bf.GetDBFI())[i]->Copy());
   }
   form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form->Assemble();

   essentialTrueDofs.Append(new Array<int>());
   spaceHierarchy.GetFESpaceAtLevel(0).GetEssentialTrueDofs(
       ess_bdr, *essentialTrueDofs.Last());

   OperatorPtr opr;
   opr.SetType(Operator::ANY_TYPE);
   form->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
   opr.SetOperatorOwner(false);

   CGSolver* pcg = new CGSolver();
   pcg->SetPrintLevel(0);
   pcg->SetMaxIter(200);
   pcg->SetRelTol(sqrt(1e-4));
   pcg->SetAbsTol(0.0);
   pcg->SetOperator(*opr.Ptr());

   AddCoarsestLevel(opr.Ptr(), pcg, true, true);

   for (int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
   {
      BilinearForm* form;
      // Reuse form on finest level
      if (level == spaceHierarchy.GetNumLevels() - 1)
      {
         form = &bf;
      }
      else
      {
         form = new BilinearForm(&spaceHierarchy.GetFESpaceAtLevel(level));
         // TODO: Copy all integrators
         for (int i = 0; i < dbfi.Size(); ++i)
         {
            form->AddDomainIntegrator((*bf.GetDBFI())[i]->Copy());
         }
         form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
         form->Assemble();
         bfs.Append(form);
      }

      essentialTrueDofs.Append(new Array<int>());
      spaceHierarchy.GetFESpaceAtLevel(level).GetEssentialTrueDofs(
          ess_bdr, *essentialTrueDofs.Last());

      OperatorPtr opr;
      opr.SetType(Operator::ANY_TYPE);
      form->FormSystemMatrix(*essentialTrueDofs.Last(), opr);
      opr.SetOperatorOwner(false);

      Vector diag(spaceHierarchy.GetFESpaceAtLevel(level).GetTrueVSize());
      form->AssembleDiagonal(diag);

      Vector ev(opr->Width());
      OperatorJacobiSmoother invDiagOperator(diag, *essentialTrueDofs.Last(),
                                             1.0);
      ProductOperator diagPrecond(&invDiagOperator, opr.Ptr(), false, false);

      PowerMethod powerMethod;
      double estLargestEigenvalue =
          powerMethod.EstimateLargestEigenvalue(diagPrecond, ev, 10, 1e-8);

      Solver* smoother = new OperatorChebyshevSmoother(
          opr.Ptr(), diag, *essentialTrueDofs.Last(), 2, estLargestEigenvalue);

      Operator* P =
          new TransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                               spaceHierarchy.GetFESpaceAtLevel(level));

      AddLevel(opr.Ptr(), smoother, P, true, true, true);
   }
}

void MultigridBilinearForm::SetupFull(SpaceHierarchy& spaceHierarchy,
                                      SparseMatrix& opr, Array<int>& ess_bdr)
{
   AddEmptyLevels(spaceHierarchy.GetNumLevels());

   width = opr.Width();
   height = opr.Height();

   operators[spaceHierarchy.GetFinestLevelIndex()] = &opr;

   for (int level = spaceHierarchy.GetFinestLevelIndex(); level > 0; --level)
   {
      smoothers[level] = new GSSmoother((SparseMatrix&)*operators[level]);

      SparseMatrix* R =
          spaceHierarchy.GetFESpaceAtLevel(level).H2L_GlobalRestrictionMatrix(
              &spaceHierarchy.GetFESpaceAtLevel(level - 1));
      SparseMatrix* P = mfem::Transpose(*R);
      prolongations[level - 1] = P;


      SparseMatrix* rap = mfem::RAP(*P, (SparseMatrix&)*operators[level], *P);

      Array<int>* ess_tdof_list = new Array<int>();
      essentialTrueDofs.Append(ess_tdof_list);
      spaceHierarchy.GetFESpaceAtLevel(level-1).GetEssentialTrueDofs(
          ess_bdr, *ess_tdof_list);

      SparseMatrix* elim = new SparseMatrix(rap->Height());

      for (int i = 0; i < ess_tdof_list->Size(); i++)
      {
         rap->EliminateRowCol((*ess_tdof_list)[i], *elim, Matrix::DiagonalPolicy::DIAG_ONE);
      }

      const int remove_zeros = 0;
      rap->Finalize(remove_zeros);

      operators[level - 1] = rap;
   }

   CGSolver* pcg = new CGSolver();
   GSSmoother* prec = new GSSmoother((SparseMatrix&)*operators[0]);
   pcg->SetPrintLevel(-1);
   pcg->SetMaxIter(50);
   pcg->SetRelTol(sqrt(1e-4));
   pcg->SetAbsTol(0.0);
   pcg->SetOperator(*operators[0]);
   pcg->SetPreconditioner(*prec);
   smoothers[0] = pcg;
}

} // namespace mfem