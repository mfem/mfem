// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license.  We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mgbilinearform.hpp"
#include "transfer.hpp"

namespace mfem
{

MultigridBilinearForm::MultigridBilinearForm() : MultigridOperator() {}

MultigridBilinearForm::MultigridBilinearForm(SpaceHierarchy& spaceHierarchy,
                                             BilinearForm& bf,
                                             Array<int>& ess_bdr,
                                             int chebyshevOrder)
   : MultigridOperator()
{
   MFEM_VERIFY(bf.GetAssemblyLevel() == AssemblyLevel::PARTIAL,
               "Assembly level must be PARTIAL");

   MFEM_VERIFY(bf.GetBBFI()->Size() == 0
               && bf.GetFBFI()->Size() == 0
               && bf.GetBFBFI()->Size() == 0,
               "Only domain integrators are currently supported");

   BilinearForm* form = new BilinearForm(&spaceHierarchy.GetFESpaceAtLevel(0));
   Array<BilinearFormIntegrator*>& dbfi = *bf.GetDBFI();
   for (int i = 0; i < dbfi.Size(); ++i)
   {
      form->AddDomainIntegrator((*bf.GetDBFI())[i]->Copy());
   }
   form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   form->Assemble();
   bfs.Append(form);

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
         opr.Ptr(), diag, *essentialTrueDofs.Last(), chebyshevOrder,
         estLargestEigenvalue);

      Operator* P =
         new TransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                              spaceHierarchy.GetFESpaceAtLevel(level));

      AddLevel(opr.Ptr(), smoother, P, true, true, true);
   }
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

} // namespace mfem
