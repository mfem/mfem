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

IncompressibleNavierSolver::IncompressibleNavierSolver(ParMesh *mesh, int velorder, int porder, int torder, real_t kin_vis)
   : pmesh(mesh), velorder(velorder), porder(porder), torder(torder), kin_vis(kin_vis),
     gll_rules(0, Quadrature1D::GaussLobatto), velGF(torder+1,nullptr), pGF(torder+1,nullptr)
{
   vfec   = new H1_FECollection(velorder, pmesh->Dimension());
   psifec = new H1_FECollection(velorder);
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

   for( int i; i<torder; i++)
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
   pfes->GetEssentialTrueDofs(pres_ess_attr, pres_ess_tdof);

   Array<int> empty;

   // GLL integration rule (Numerical Integration)
   const IntegrationRule &ir_ni = gll_rules.Get(vfes->GetFE(0)->GetGeomType(),
                                                2 * velorder - 1);

   kinvisCoeff  = new ConstantCoefficient(kin_vis);
   dtCoeff      = new ConstantCoefficient(-1.0/dt);

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
   pUnitVectorCoeff = new UnitVectorGridFunctionCoeff(pmesh->Dimension());
   nonlinTermCoeff = new NonLinTermVectorGridFunctionCoeff(pmesh->Dimension());
   auto *pvel_lfi = new VectorDomainLFGradIntegrator(*pUnitVectorCoeff);
   auto *p_nonlintermlfi = new VectorDomainLFIntegrator(*nonlinTermCoeff);
   if (numerical_integ)
   {
      pvel_lfi->SetIntRule(&ir_ni);
      p_nonlintermlfi->SetIntRule(&ir_ni);
   }
   velLForm->AddDomainIntegrator(pvel_lfi);
   velLForm->AddDomainIntegrator(p_nonlintermlfi);

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

   velInvPC = new HypreSmoother(*vOp.As<HypreParMatrix>());
   dynamic_cast<HypreSmoother *>(velInvPC)->SetType(HypreSmoother::Jacobi, 1);

   velInv = new CGSolver(vfes->GetComm());
   velInv->iterative_mode = true;
   velInv->SetOperator(*vOp);
   velInv->SetPreconditioner(*velInvPC);
   velInv->SetPrintLevel(pl_velsolve);
   velInv->SetRelTol(rtol_velsolve);
   velInv->SetMaxIter(200);

   psiInvPC = new HypreSmoother(*psiOp.As<HypreParMatrix>());
   dynamic_cast<HypreSmoother *>(psiInvPC)->SetType(HypreSmoother::Jacobi, 1);

   psiInv = new CGSolver(vfes->GetComm());
   psiInv->iterative_mode = true;
   psiInv->SetOperator(*psiOp);
   psiInv->SetPreconditioner(*psiInvPC);
   psiInv->SetPrintLevel(pl_psisolve);
   psiInv->SetRelTol(rtol_psisolve);
   psiInv->SetMaxIter(200);

   pInvPC = new HypreSmoother(*pOp.As<HypreParMatrix>());
   dynamic_cast<HypreSmoother *>(pInvPC)->SetType(HypreSmoother::Jacobi, 1);

   pInv = new CGSolver(vfes->GetComm());
   pInv->iterative_mode = true;
   pInv->SetOperator(*pOp);
   pInv->SetPreconditioner(*pInvPC);
   pInv->SetPrintLevel(pl_psolve);
   pInv->SetRelTol(rtol_psolve);
   pInv->SetMaxIter(200);

}

void IncompressibleNavierSolver::UpdateTimestepHistory(real_t dt)
{

}

void IncompressibleNavierSolver::Step(real_t &time, real_t dt, int current_step,
                        bool provisional)
{
   pUnitVectorCoeff->SetGridFunction( pGF[0] );
   nonlinTermCoeff->SetGridFunction( velGF[1] );



   subtract(1.0/dt, *velGF[1], *velGF[0], DvGF);
   DvelCoeff->SetGridFunction( &DvGF );




   divVelCoeff->SetGridFunction( velGF[0]);
   divVelGF.ProjectCoefficient( *divVelCoeff );

   add( *pGF[1], psiGF, pRHS);
   add( pRHS, -1.0*kin_vis, divVelGF, pRHS);
   pRHSCoeff->SetGridFunction( &pRHS );

   //-------------------------------------------------------------------------

   Array<int> empty;
   // velBForm->Update();
   // velBForm->Assemble();
   // velBForm->FormSystemMatrix(vel_ess_tdof, vOp);

   // psiBForm->Update();
   // psiBForm->Assemble();
   // psiBForm->FormSystemMatrix(vel_ess_tdof, psiOp);

   // vpBForm->Update();
   // vpBForm->Assemble();
   // vpBForm->FormSystemMatrix(vel_ess_tdof, pOp);

   //-------------------------------------------------------------------------
   velLForm->Assemble();
   velLForm->ParallelAssemble(velLF);

   psiLForm->Assemble();
   psiLForm->ParallelAssemble(psiLF);

   pLForm->Assemble();
   pLForm->ParallelAssemble(pLF);

   //-------------------------------------------------------------------------

   Vector X1, B1;
   Vector X2, B2;
   Vector X3, B3;
   if (partial_assembly)
   {

   }
   else
   {
      velBForm->FormLinearSystem(vel_ess_tdof, *velGF[0], velLF, vOp  , X1, B1, 1);
      psiBForm->FormLinearSystem(empty       , psiGF    , psiLF, psiOp, X2, B2, 1);
      pBForm  ->FormLinearSystem(empty        , *pGF[0] , pLF  , pOp  , X3, B3, 1);
   }

   velInv->Mult(B1, X1);
   iter_vsolve = velInv->GetNumIterations();
   res_vsolve = velInv->GetFinalNorm();
   velBForm->RecoverFEMSolution(X1, velLF, *velGF[0]);

   psiInv->Mult(B2, X2);
   iter_psisolve = psiInv->GetNumIterations();
   res_psisolve = psiInv->GetFinalNorm();
   psiBForm->RecoverFEMSolution(X2, psiLF, psiGF);

   pInv->Mult(B3, X3);
   iter_psolve = pInv->GetNumIterations();
   res_psolve = pInv->GetFinalNorm();
   pBForm->RecoverFEMSolution(X3, pLF, *pGF[0]);

   *velGF[1] = *velGF[0];
   *pGF[1]   = *pGF[0];



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

}

real_t IncompressibleNavierSolver::ComputeCFL(ParGridFunction &u, real_t dt)
{
   
   return 0;
}

void IncompressibleNavierSolver::AddVelDirichletBC(VectorCoefficient *coeff, Array<int> &attr)
{

}

void IncompressibleNavierSolver::AddVelDirichletBC(VecFuncT *f, Array<int> &attr)
{

}

IncompressibleNavierSolver::~IncompressibleNavierSolver()
{

   delete velBForm;
   delete psiBForm;
   delete pBForm;

   delete kinvisCoeff;
   delete dtCoeff;

   for( int i; i<torder; i++)
   {
      delete velGF[i];
      delete pGF[i];
   }

   delete DvelCoeff;
   delete divVelCoeff;
   delete pRHSCoeff;
   delete pUnitVectorCoeff;

   delete vfec;
   delete psifec;
   delete pfec;
   delete vfes;
   delete psifes;
   delete pfes;
}
