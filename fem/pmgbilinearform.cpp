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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pmgbilinearform.hpp"
#include "transfer.hpp"

namespace mfem
{

ParMultigridBilinearForm::ParMultigridBilinearForm(ParSpaceHierarchy& spaceHierarchy,
                                             ParBilinearForm& bf,
                                             Array<int>& ess_bdr)
    : MultigridBilinearForm()
{
   MFEM_VERIFY(bf.GetAssemblyLevel() == AssemblyLevel::PARTIAL,
               "Assembly level must be PARTIAL");

   ParBilinearForm* form = new ParBilinearForm(&spaceHierarchy.GetFESpaceAtLevel(0));
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

   CGSolver* pcg = new CGSolver(MPI_COMM_WORLD);
   pcg->SetPrintLevel(0);
   pcg->SetMaxIter(200);
   pcg->SetRelTol(sqrt(1e-4));
   pcg->SetAbsTol(0.0);
   pcg->SetOperator(*opr.Ptr());

   AddCoarsestLevel(opr.Ptr(), pcg, true, true);

   // ParMesh* pmesh = spaceHierarchy.GetFESpaceAtLevel(0).GetParMesh();
   // ParMesh* pmesh_lor = new ParMesh(pmesh, 1, BasisType::GaussLobatto);
   // H1_FECollection* fec_lor = new H1_FECollection(1, pmesh->Dimension(),
   //                                  BasisType::GaussLobatto);
   // ParFiniteElementSpace* fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);
   // ParBilinearForm* a_pc = new ParBilinearForm(fespace_lor);

   // Array<BilinearFormIntegrator*>& dbfi = *bf.GetDBFI();
   // for (int i = 0; i < dbfi.Size(); ++i)
   // {
   //    a_pc->AddDomainIntegrator((*bf.GetDBFI())[i]->Copy());
   // }

   // a_pc->UsePrecomputedSparsity();
   // a_pc->Assemble();

   // essentialTrueDofs.Append(new Array<int>());
   // spaceHierarchy.GetFESpaceAtLevel(0).GetEssentialTrueDofs(
   //     ess_bdr, *essentialTrueDofs.Last());

   // HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
   // a_pc->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

   // HypreBoomerAMG* amg = new HypreBoomerAMG(*hypreCoarseMat);
   // amg->SetPrintLevel(-1);
   // amg->SetMaxIter(1);

   // AddCoarsestLevel(hypreCoarseMat, amg, true, true);

   for (int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
   {
      ParBilinearForm* form;
      // Reuse form on finest level
      if (level == spaceHierarchy.GetNumLevels() - 1)
      {
         form = &bf;
      }
      else
      {
         form = new ParBilinearForm(&spaceHierarchy.GetFESpaceAtLevel(level));
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

      PowerMethod powerMethod(MPI_COMM_WORLD);
      double estLargestEigenvalue =
          powerMethod.EstimateLargestEigenvalue(diagPrecond, ev, 10, 1e-8);

      Solver* smoother = new OperatorChebyshevSmoother(
          opr.Ptr(), diag, *essentialTrueDofs.Last(), 2, estLargestEigenvalue);

      Operator* P =
          new TrueTransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                               spaceHierarchy.GetFESpaceAtLevel(level));

      AddLevel(opr.Ptr(), smoother, P, true, true, true);
   }
}

ParMultigridBilinearForm::~ParMultigridBilinearForm()
{}

}
#endif