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

#include <memory>

#include "mtop_solvers.hpp"

using namespace mfem;

StokesSolver::StokesSolver(ParMesh* mesh, int order_, bool zero_mean_press_):
   pmesh(mesh),
   order(order_),
   dim(mesh->SpaceDimension()),
   zero_mean_press(zero_mean_press_)
{
   if (order_<2) { order=2;}

   vfec=new H1_FECollection(order, pmesh->Dimension());
   pfec=new H1_FECollection(order-1, pmesh->Dimension());
   vfes=new ParFiniteElementSpace(pmesh, vfec, pmesh->Dimension());
   pfes=new ParFiniteElementSpace(pmesh, pfec);

   vel.SetSpace(vfes); vel=0.0;
   pre.SetSpace(pfes); pre=0.0;

   avel.SetSpace(vfes); avel=0.0;
   apre.SetSpace(pfes); apre=0.0;

   brink.reset();
   visc.reset(new ConstantCoefficient(0.001));

   onecoeff.constant = 1.0;
   zerocoef.constant = 0.0;

   siz_u=vfes->TrueVSize();
   siz_p=pfes->TrueVSize();

   block_true_offsets.SetSize(3);
   block_true_offsets[0] = 0;
   block_true_offsets[1] = siz_u;
   block_true_offsets[2] = siz_p;
   block_true_offsets.PartialSum();
   //set the width and the height of the operator
   this->width=  block_true_offsets[2];
   this->height= block_true_offsets[2];


   sol.Update(block_true_offsets); sol=0.0;
   rhs.Update(block_true_offsets); rhs=0.0;
   adj.Update(block_true_offsets); adj=0.0;

   ess_tdofv.SetSize(0);

   bf11.reset();
   bf12.reset();
   bf21.reset();

   SetLinearSolver();
}

StokesSolver::~StokesSolver()
{
   delete pfes;
   delete vfes;
   delete pfec;
   delete vfec;
}

void StokesSolver::SetEssTDofsV(mfem::Array<int>& ess_dofs)
{
   // Set the essential boundary conditions
   ess_dofs.DeleteAll();

   Array<int> ess_bdr(pmesh->bdr_attributes.Max());
   ess_bdr=0;
   for (auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      ess_bdr[attr-1] = 1;
   }
   vfes->GetEssentialTrueDofs(ess_bdr,ess_dofs);
}

void StokesSolver::SetEssTDofsV(Vector& v) const
{
   for (auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      std::shared_ptr<VectorCoefficient> coeff = it->second;
      coeff->SetTime(real_t(0.0));

      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[attr-1] = 1;

      mfem::Array<int> loc_tdofs;
      vfes->GetEssentialTrueDofs(ess_bdr,loc_tdofs);
      vel.ProjectBdrCoefficient(*coeff,ess_bdr);
      vel.SetTrueVector();

      // copy values to v
      Vector &tvel=vel.GetTrueVector();
      //vel.GetTrueDofs(tvel);
      for (int j=0; j<loc_tdofs.Size(); j++)
      {
         v[loc_tdofs[j]]=tvel[loc_tdofs[j]];
      }
   }
}

void StokesSolver::SetEssVBC(ParGridFunction& pgf)
{

   for (auto it=vel_bcs.begin(); it!=vel_bcs.end(); ++it)
   {
      int attr = it->first;
      std::shared_ptr<VectorCoefficient> coeff = it->second;
      coeff->SetTime(real_t(0.0));
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr=0;
      ess_bdr[attr-1] = 1;

      pgf.ProjectBdrCoefficient(*coeff,ess_bdr);
   }
}

void StokesSolver::DeleteBC()
{
   vel_bcs.clear();
   ess_tdofv.DeleteAll();
   //delete allocated matrices and forms

   bf11.reset();
   bf12.reset();
   bf21.reset();

   A11.reset();
   A12.reset();
   A21.reset();

   A11e.reset();
   A12e.reset();
   A21e.reset();

   bop.reset();
   ls.reset();
   prec.reset();
}

void StokesSolver::Assemble()
{
   //set BC
   vel=real_t(0.0);
   pre=real_t(0.0);
   SetEssVBC(vel);
   SetEssTDofsV(ess_tdofv);

   //assemble block 11
   bf11.reset(new ParBilinearForm(vfes));
   bf11->AddDomainIntegrator(new ElasticityIntegrator(zerocoef,*visc));
   if (nullptr!=brink.get())
   {
      bf11->AddDomainIntegrator(new VectorMassIntegrator(*brink));
   }
   bf11->Assemble(0);
   bf11->Finalize(0);
   A11.reset(bf11->ParallelAssemble());

   //assemble block 12
   bf12.reset(new ParMixedBilinearForm(pfes, vfes));
   //bf12->AddDomainIntegrator(new GradientIntegrator());
   bf12->AddDomainIntegrator(
               new TransposeIntegrator(
                   new VectorDivergenceIntegrator()));
   bf12->Assemble(0);
   bf12->Finalize(0);
   A12.reset(bf12->ParallelAssemble());

   //assemble block 21
   bf21.reset(new ParMixedBilinearForm(vfes, pfes));
   bf21->AddDomainIntegrator(
               new VectorDivergenceIntegrator());
   bf21->Assemble(0);
   bf21->Finalize(0);
   A21.reset(bf21->ParallelAssemble());

   //set BC to the operators
   A11e.reset(A11->EliminateRowsCols(ess_tdofv));
   A12->EliminateRows(ess_tdofv);
   A21e.reset(A21->EliminateCols(ess_tdofv));

   //set the block operator
   bop.reset(new BlockOperator(block_true_offsets));
   bop->SetBlock(0,0,A11.get());
   bop->SetBlock(0,1,A12.get());
   bop->SetBlock(1,0,A21.get());

   if (zero_mean_press)
   {
      V.SetSize(pfes->GetTrueVSize()); V=0.0;
      ParLinearForm lf(pfes);
      lf.AddDomainIntegrator(new DomainLFIntegrator(onecoeff));
      lf.Assemble();
      lf.ParallelAssemble(V);
   }

   //set the solver to GMRES
   {
      //GMRESSolver* gmres=new GMRESSolver(pmesh->GetComm());

      //MINRESSolver* gmres=new MINRESSolver(pmesh->GetComm());
      FGMRESSolver* gmres=new FGMRESSolver(pmesh->GetComm());
      gmres->SetKDim(100);
      gmres->SetRelTol(linear_rtol);
      gmres->SetAbsTol(linear_atol);
      gmres->SetMaxIter(linear_iter);
      gmres->SetOperator(*bop);
      gmres->SetPrintLevel(1);

      //prec.reset(new DLSCPrec(A11.get(),A21.get(),A12.get(), zero_mean_press));

      LSCStokesPrec* lsc=new LSCStokesPrec(vfes,pfes,visc,brink,ess_tdofv,
                                           A11.get(),A12.get(),A21.get(),zero_mean_press);

      prec.reset(lsc);
      prec->SetMaxIter(100);
      prec->SetAbsTol(1e-12);
      prec->SetRelTol(1e-5);
      gmres->SetPreconditioner(*prec);

      ls.reset(gmres);
   }

}

void StokesSolver::FSolve()
{
   Vector& vsol=sol.GetBlock(0);
   Vector& psol=sol.GetBlock(1);

   Vector& vrhs=rhs.GetBlock(0);
   Vector& prhs=rhs.GetBlock(1);

   //assemble the RHS
   rhs=0.0;
   if (nullptr!=vol_force.get())
   {
      ParLinearForm lf(vfes);
      lf.AddDomainIntegrator(new VectorDomainLFIntegrator(*vol_force));
      lf.Assemble();
      lf.ParallelAssemble(vrhs);
   }

   //set the velocity BCs
   SetEssTDofsV(vsol);


   //modify the rhs
   A21e->Mult(-1.0,vsol,1.0,prhs);
   A11->EliminateBC(*A11e,ess_tdofv,vsol,vrhs);

   //solve the linear system
   ls->Mult(rhs,sol);

}

void StokesSolver::ASolve(mfem::Vector &rhs)
{
   MultTranspose(rhs,adj);
}

void StokesSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   //copy x to rhs
   {
      int N = x.Size();
      const real_t *xp = x.Read();
      real_t *rp = rhs.ReadWrite();
      forall(N, [=] MFEM_HOST_DEVICE(int i) { rp[i] = xp[i]; });
   }

   BlockVector yb(y, block_true_offsets);
   Vector& vsol=yb.GetBlock(0);
   Vector& psol=yb.GetBlock(1);

   Vector& vrhs=rhs.GetBlock(0);
   Vector& prhs=rhs.GetBlock(1);

   //set the velocity BCs
   SetEssTDofsV(vsol);

   //modify the rhs
   A21e->Mult(-1.0,vsol,1.0,prhs);
   A11->EliminateBC(*A11e,ess_tdofv,vsol,vrhs);

   //solve the linear system
   ls->Mult(rhs,yb);
}

void StokesSolver::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const
{
   //copy x to rhs
   {
      int N = x.Size();
      const real_t *xp = x.Read();
      real_t *rp = rhs.ReadWrite();
      forall(N, [=] MFEM_HOST_DEVICE(int i) { rp[i] = xp[i]; });
   }

   BlockVector yb(y, block_true_offsets);
   Vector& vsol=yb.GetBlock(0);
   Vector& psol=yb.GetBlock(1);

   Vector& vrhs=rhs.GetBlock(0);
   Vector& prhs=rhs.GetBlock(1);

   //set zero velocity bc
   {
      int N = ess_tdofv.Size();
      real_t *yp = vsol.ReadWrite();
      const int *ep = ess_tdofv.Read();
      forall(N, [=] MFEM_HOST_DEVICE(int i) { yp[ep[i]] = 0.0; });
   }

   //modify the rhs
   //A21e->Mult(-1.0,vsol,1.0,prhs); // vsol is zero at the BC
   A11->EliminateBC(*A11e,ess_tdofv,vsol,vrhs);

   //solve the linear system
   ls->Mult(rhs,yb);
}
