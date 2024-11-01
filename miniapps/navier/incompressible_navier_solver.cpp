// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "incompressible_navier_solver.hpp"
#include "../../general/forall.hpp"
#include <fstream>
#include <iomanip>

using namespace mfem;
using namespace incompressible_navier;

IncompressibleNavierSolver::IncompressibleNavierSolver(ParMesh *mesh, int velorder, int porder, int torder_, real_t kin_vis)
   : pmesh(mesh), velorder(velorder), porder(porder), torder(torder_), kin_vis(kin_vis),
     gll_rules(0, Quadrature1D::GaussLobatto), velGF(torder_+1,nullptr), pGF(torder_+1,nullptr)
{
   vfec   = new H1_FECollection(velorder, pmesh->Dimension());
   psifec = new H1_FECollection(porder);
   pfec   = new H1_FECollection(porder);
   vfes   = new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
   psifes = new ParFiniteElementSpace(pmesh, pfec);
   pfes   = new ParFiniteElementSpace(pmesh, pfec);

   // Check if fully periodic mesh
   if (!(pmesh->bdr_attributes.Size() == 0))
   {
      vel_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      vel_ess_attr = 0;

      pres_ess_attr.SetSize(pmesh->bdr_attributes.Max());
      pres_ess_attr = 0;
   }

   int vfes_truevsize = vfes->GetTrueVSize();
   int pfes_truevsize = pfes->GetTrueVSize();

   for( int i = 0; i<torder+1; i++)
   {
      velGF[i] = new ParGridFunction(vfes); *velGF[i] = 0.0;
      pGF[i]   = new ParGridFunction(pfes); *pGF[i]   = 0.0;
   }

   psiGF.SetSpace(psifes);
   DvGF.SetSpace(vfes);
   divVelGF.SetSpace(pfes);
   pRHS.SetSpace(pfes);
}

void IncompressibleNavierSolver::Setup(real_t dt)
{
   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Setup" << std::endl;
      if (partial_assembly)
      {
         mfem::out << "Using Partial Assembly" << std::endl;
      }
      else
      {
         mfem::out << "Using Full Assembly" << std::endl;
      }
   }

   vfes->GetEssentialTrueDofs(vel_ess_attr, vel_ess_tdof);
   //pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);

   Array<int> empty;

   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(),
                                                2 * velorder - 1);

   kinvisCoeff  = new ConstantCoefficient(kin_vis);
   dtCoeff      = new ConstantCoefficient(1.0/dt);

   //-------------------------------------------------------------------------

   velBForm = new ParBilinearForm(vfes);
   auto *vmass_blfi = new VectorMassIntegrator(*dtCoeff);       
   auto *vdiff_blfi = new VectorDiffusionIntegrator(*kinvisCoeff);   

   if (numerical_integ)
   {
       vmass_blfi->SetIntRule(&ir_ni); 
       vdiff_blfi->SetIntRule(&ir_ni);
   }
   velBForm->AddDomainIntegrator(vmass_blfi);
   velBForm->AddDomainIntegrator(vdiff_blfi);
   if (partial_assembly)
   {
      velBForm->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   velBForm->Assemble();
   velBForm->FormSystemMatrix(vel_ess_tdof, vOp);

   //-------------------------------------------------------------------------

   psiBForm = new ParBilinearForm(psifes);
   auto *psidiff_blfi = new DiffusionIntegrator;

   if (numerical_integ)
   {
       psidiff_blfi->SetIntRule(&ir_ni); 
   }
   psiBForm->AddDomainIntegrator(psidiff_blfi);
   if (partial_assembly)
   {
      psiBForm->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   psiBForm->Assemble();
   psiBForm->FormSystemMatrix(empty, psiOp);

   //-------------------------------------------------------------------------

   pBForm = new ParBilinearForm(pfes);
   auto *pmass_blfi = new MassIntegrator;

   if (numerical_integ)
   {
       pmass_blfi->SetIntRule(&ir_ni); 
   }
   pBForm->AddDomainIntegrator(pmass_blfi);
   if (partial_assembly)
   {
      pBForm->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }

   pBForm->Assemble();
   pBForm->FormSystemMatrix(empty, pOp);

   //-------------------------------------------------------------------------

   velLForm = new ParLinearForm(vfes);
   prevVelLoadCoeff = new PrevVelVectorGridFunctionCoeff(pmesh->Dimension());
   pUnitVectorCoeff = new UnitVectorGridFunctionCoeff(pmesh->Dimension());
   nonlinTermCoeff = new NonLinTermVectorGridFunctionCoeff(pmesh->Dimension());
   auto *prevVelLoadLFi = new VectorDomainLFIntegrator(*prevVelLoadCoeff);
   auto *pvel_lfi = new VectorDomainLFGradIntegrator(*pUnitVectorCoeff);
   auto *p_nonlintermlfi = new VectorDomainLFIntegrator(*nonlinTermCoeff);
   if (numerical_integ)
   {
      prevVelLoadLFi->SetIntRule(&ir_ni);
      pvel_lfi->SetIntRule(&ir_ni);
      p_nonlintermlfi->SetIntRule(&ir_ni);
   }
   velLForm->AddDomainIntegrator(prevVelLoadLFi);
   velLForm->AddDomainIntegrator(pvel_lfi);
   velLForm->AddDomainIntegrator(p_nonlintermlfi);

   // for (auto &vel_dbc : vel_dbcs)
   // {
   //    auto *bdr_integrator = new BoundaryNormalLFIntegrator(*vel_dbc.coeff);
   //    if (numerical_integ)
   //    {
   //       bdr_integrator->SetIntRule(&ir_ni);
   //    }
   //    velLForm->AddBoundaryIntegrator(bdr_integrator, vel_dbc.attr);
   // }

   //-------------------------------------------------------------------------

   psiLForm = new ParLinearForm(psifes);
   DvelCoeff = new VectorGridFunctionCoefficient;
   auto *Dvel_lfi = new DomainLFGradIntegrator(*DvelCoeff);
   if (numerical_integ)
   {
      Dvel_lfi->SetIntRule(&ir_ni);
   }
   psiLForm->AddDomainIntegrator(Dvel_lfi);

   //-------------------------------------------------------------------------

   pLForm = new ParLinearForm(pfes); 
   divVelCoeff = new DivergenceGridFunctionCoefficient(velGF[0]);
   pRHSCoeff = new GridFunctionCoefficient(&pRHS);
   auto *p_lfi = new DomainLFIntegrator(*pRHSCoeff);
   if (numerical_integ)
   {
      p_lfi->SetIntRule(&ir_ni);
   }
   pLForm->AddDomainIntegrator(p_lfi);

   //-------------------------------------------------------------------------

   if (partial_assembly)
   {
      Vector diag_pa(vfes->GetTrueVSize());
      velBForm->AssembleDiagonal(diag_pa);
      velInvPC = new OperatorJacobiSmoother(diag_pa, vel_ess_tdof);
   }
   else
   {
      velInvPC = new HypreSmoother(*vOp.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(velInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }

   velInv = new CGSolver(vfes->GetComm());
   velInv->iterative_mode = true;
   velInv->SetOperator(*vOp);
   velInv->SetPreconditioner(*velInvPC);
   velInv->SetPrintLevel(pl_velsolve);
   velInv->SetRelTol(rtol_velsolve);
   velInv->SetMaxIter(1200);

   if (partial_assembly)
   {
      int psifes_truevsize = psifes->GetTrueVSize();
      mfem::Vector psin(psifes_truevsize);      psin = 0.0;
      mfem::Vector respsi(psifes_truevsize);    respsi = 0.0;

      lor = new ParLORDiscretization(*psiBForm, empty);
      psiInvPC = new HypreBoomerAMG(lor->GetAssembledMatrix());
      psiInvPC->SetPrintLevel(0);
      psiInvPC->Mult(respsi, psin);
      SpInvOrthoPC = new OrthoSolver(psifes->GetComm());
      SpInvOrthoPC->SetSolver(*psiInvPC);
   }
   else
   {
      psiInvPC = new HypreBoomerAMG(*psiOp.As<HypreParMatrix>());
      psiInvPC->SetPrintLevel(0);
      SpInvOrthoPC = new OrthoSolver(psifes->GetComm());
      SpInvOrthoPC->SetSolver(*psiInvPC);
   }

   psiInv = new CGSolver(psifes->GetComm());
   psiInv->iterative_mode = true;
   psiInv->SetOperator(*psiOp);
   psiInv->SetPreconditioner(*SpInvOrthoPC);
   psiInv->SetPrintLevel(pl_psisolve);
   psiInv->SetRelTol(rtol_psisolve);
   psiInv->SetMaxIter(1000);

   if (partial_assembly)
   {
      Vector diag_pa(pfes->GetTrueVSize());
      pBForm->AssembleDiagonal(diag_pa);
      pInvPC = new OperatorJacobiSmoother(diag_pa, empty);
   }
   else
   {
      pInvPC = new HypreSmoother(*pOp.As<HypreParMatrix>());
      dynamic_cast<HypreSmoother *>(pInvPC)->SetType(HypreSmoother::Jacobi, 1);
   }

   pInv = new CGSolver(pfes->GetComm());
   pInv->iterative_mode = true;
   pInv->SetOperator(*pOp);
   pInv->SetPreconditioner(*pInvPC);
   pInv->SetPrintLevel(pl_psolve);
   pInv->SetRelTol(rtol_psolve);
   pInv->SetMaxIter(1000);
}

void IncompressibleNavierSolver::UpdateTimestepHistory(real_t dt)
{

}

void IncompressibleNavierSolver::Step(real_t &time, real_t dt, int current_step,
                        bool provisional)
{
   for (auto &vel_dbc : vel_dbcs)
   {
      velGF[0]->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
      velGF[1]->ProjectBdrCoefficient(*vel_dbc.coeff, vel_dbc.attr);
   }

   //-------------------------------------------------------------------------

   prevVelLoadCoeff ->SetGridFunction( velGF[1], dt ); 
   pUnitVectorCoeff->SetGridFunction( pGF[1] );
   nonlinTermCoeff->SetGridFunction( velGF[1] );

   Array<int> empty;

   //-------------------------------------------------------------------------

   velLForm->Assemble();
   velLForm->ParallelAssemble(velLF);

   //-------------------------------------------------------------------------

   Vector X1, B1;
   Vector X2, B2;
   Vector X3, B3;
   if (partial_assembly)
   {
      auto *vpC = vOp.As<ConstrainedOperator>();
      EliminateRHS(*velBForm, *vpC, vel_ess_tdof, *velGF[0], velLF, X1, B1, 1);
   }
   else
   {
      velBForm->FormLinearSystem(vel_ess_tdof, *velGF[0], velLF, vOp  , X1, B1, 1);
   }

   velInv->Mult(B1, X1);
   iter_vsolve = velInv->GetNumIterations();
   res_vsolve = velInv->GetFinalNorm();
   velBForm->RecoverFEMSolution(X1, velLF, *velGF[0]);

   //-------------------------------------------------------------------------
   
   subtract(1.0/dt, *velGF[0], *velGF[1], DvGF);
   DvelCoeff->SetGridFunction( &DvGF );

   psiLForm->Assemble();
   psiLForm->ParallelAssemble(psiLF);

   if (partial_assembly)
   {
      auto *psipC = psiOp.As<ConstrainedOperator>();
      EliminateRHS(*psiBForm, *psipC, empty, psiGF, psiLF, X2, B2, 1);
   }
   else
   {
      psiBForm->FormLinearSystem(empty, psiGF, psiLF, psiOp, X2, B2, 1);
   }

   psiInv->Mult(B2, X2);
   iter_psisolve = psiInv->GetNumIterations();
   res_psisolve = psiInv->GetFinalNorm();
   psiBForm->RecoverFEMSolution(X2, psiLF, psiGF);

   //-------------------------------------------------------------------------
   divVelCoeff->SetGridFunction( velGF[0]);
   divVelGF.ProjectCoefficient( *divVelCoeff );

   add( *pGF[1], psiGF, pRHS);
   add( pRHS, -1.0*kin_vis, divVelGF, pRHS);
   pRHSCoeff->SetGridFunction( &pRHS );

   pLForm->Assemble();
   pLForm->ParallelAssemble(pLF);

   if (partial_assembly)
   {
      auto *ppC = pOp.As<ConstrainedOperator>();
      EliminateRHS(*pBForm, *ppC, empty, *pGF[0], pLF, X3, B3, 1);
   }
   else
   {
      pBForm->FormLinearSystem(empty, *pGF[0] , pLF , pOp , X3, B3, 1);
   }

   pInv->Mult(B3, X3);
   iter_psolve = pInv->GetNumIterations();
   res_psisolve = pInv->GetFinalNorm();
   pBForm->RecoverFEMSolution(X3, pLF, *pGF[0]);

   mfem::out << "It: " << iter << " | Iter_U: " << iter_vsolve << " | Iter_Psi: " << iter_psisolve << " | Iter_P: " << iter_psolve  << "\n";
   mfem::out << "It: " << iter << " | Resid_U: " << res_vsolve << " | Resid_Psi: " << res_psisolve << " | Resid_P: " << res_psisolve  << "\n";

   *velGF[1] = *velGF[0];
   *pGF[1]   = *pGF[0];

   time += dt;
   iter ++;
}



void IncompressibleNavierSolver::EliminateRHS(Operator &A,
                                ConstrainedOperator &constrainedA,
                                const Array<int> &ess_tdof_list,
                                Vector &x,
                                Vector &b,
                                Vector &X,
                                Vector &B,
                                int copy_interior)
{
   const Operator *Po = A.GetOutputProlongation();
   const Operator *Pi = A.GetProlongation();
   const Operator *Ri = A.GetRestriction();
   A.InitTVectors(Po, Ri, Pi, x, b, X, B);
   if (!copy_interior)
   {
      X.SetSubVectorComplement(ess_tdof_list, 0.0);
   }
   constrainedA.EliminateRHS(X, B);
}

real_t IncompressibleNavierSolver::ComputeCFL(ParGridFunction &u, real_t dt)
{
   
   return 0;
}

void IncompressibleNavierSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{
   vel_dbcs.emplace_back(attr, coeff);

   if (verbose && pmesh->GetMyRank() == 0)
   {
      mfem::out << "Adding Velocity Dirichlet BC to attributes ";
      for (int i = 0; i < attr.Size(); ++i)
      {
         if (attr[i] == 1)
         {
            mfem::out << i << " ";
         }
      }
      mfem::out << std::endl;
   }

   for (int i = 0; i < attr.Size(); ++i)
   {
      MFEM_ASSERT((vel_ess_attr[i] && attr[i]) == 0,
                  "Duplicate boundary definition deteceted.");
      if (attr[i] == 1)
      {
         vel_ess_attr[i] = 1;
      }
   }

}

void IncompressibleNavierSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{
   AddVelDirichletBC(new VectorFunctionCoefficient(pmesh->Dimension(), f), attr);
}

IncompressibleNavierSolver::~IncompressibleNavierSolver()
{
   delete velBForm;
   delete psiBForm;
   delete pBForm;

   delete kinvisCoeff;
   delete dtCoeff;

   for( int i = 0; i<torder+1; i++)
   {
      delete velGF[i];
      delete pGF[i];
   }

   delete DvelCoeff;
   delete divVelCoeff;
   delete pRHSCoeff;
   delete pUnitVectorCoeff;

   delete velInv;
   delete velInvPC;
   delete psiInv;
   delete SpInvOrthoPC;
   delete psiInvPC;
   delete lor;
   delete pInv;
   delete pInvPC;

   delete vfec;
   delete psifec;
   delete pfec;
   delete vfes;
   delete psifes;
   delete pfes;
}
