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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "pmgbilinearform.hpp"
#include "transfer.hpp"

namespace mfem
{

ParMultigridBilinearForm::ParMultigridBilinearForm(
   ParSpaceHierarchy& spaceHierarchy, ParBilinearForm& bf, Array<int>& ess_bdr,
   int chebyshevOrder)
   : MultigridBilinearForm()
{
   MFEM_VERIFY(bf.GetAssemblyLevel() == AssemblyLevel::PARTIAL,
               "Assembly level must be PARTIAL");

   MFEM_VERIFY(bf.GetBBFI()->Size() == 0
               && bf.GetFBFI()->Size() == 0
               && bf.GetBFBFI()->Size() == 0,
               "Only domain integrators are currently supported");

   ParMesh* pmesh = spaceHierarchy.GetFESpaceAtLevel(0).GetParMesh();
   pmesh_lor = new ParMesh(pmesh, 1, BasisType::GaussLobatto);
   fec_lor =
      new H1_FECollection(1, pmesh->Dimension(), BasisType::GaussLobatto);
   fespace_lor = new ParFiniteElementSpace(pmesh_lor, fec_lor);
   a_lor = new ParBilinearForm(fespace_lor);

   Array<BilinearFormIntegrator*>& dbfi = *bf.GetDBFI();
   for (int i = 0; i < dbfi.Size(); ++i)
   {
      a_lor->AddDomainIntegrator((*bf.GetDBFI())[i]->Copy());
   }

   a_lor->UsePrecomputedSparsity();
   a_lor->Assemble();

   essentialTrueDofs.Append(new Array<int>());
   spaceHierarchy.GetFESpaceAtLevel(0).GetEssentialTrueDofs(
      ess_bdr, *essentialTrueDofs.Last());

   HypreParMatrix* hypreCoarseMat = new HypreParMatrix();
   a_lor->FormSystemMatrix(*essentialTrueDofs.Last(), *hypreCoarseMat);

   HypreBoomerAMG* amg = new HypreBoomerAMG(*hypreCoarseMat);
   amg->SetPrintLevel(-1);
   amg->SetMaxIter(1);

   AddCoarsestLevel(hypreCoarseMat, amg, true, true);

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
         opr.Ptr(), diag, *essentialTrueDofs.Last(), chebyshevOrder,
         estLargestEigenvalue);

      Operator* P =
         new TrueTransferOperator(spaceHierarchy.GetFESpaceAtLevel(level - 1),
                                  spaceHierarchy.GetFESpaceAtLevel(level));

      AddLevel(opr.Ptr(), smoother, P, true, true, true);
   }
}

ParMultigridBilinearForm::~ParMultigridBilinearForm()
{
   delete a_lor;
   delete fespace_lor;
   delete fec_lor;
   delete pmesh_lor;
}

} // namespace mfem
#endif
