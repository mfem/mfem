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

StokesSolver::StokesSolver(ParMesh* mesh, int order_, int num_mesh_ref_):
   pmesh(mesh),
   order(order_),
   num_mesh_ref(num_mesh_ref_),
   dim(mesh->SpaceDimension())
{
   if (order_<2) { order=2;}

   {
       meshes.Append(pmesh);
       // uniform refinement of the mesh
       for(int i=0;i<num_mesh_ref;i++){
           ParMesh* nmesh=new ParMesh(*(meshes.Last()));
           nmesh->UniformRefinement();
           meshes.Append(nmesh);
       }

       pmesh=meshes.Last();

       vfec=new H1_FECollection(order, dim);
       pfec=new H1_FECollection(order-1, dim);

       //construct the FEM spaces
       vfes= new ParFiniteElementSpace(pmesh, vfec, dim, Ordering::byNODES);
       pfes=new ParFiniteElementSpace(pmesh, pfec);
   }

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

   MPI_Comm_rank(pmesh->GetComm(),&myrank);

   sol_method=1;
}

StokesSolver::~StokesSolver()
{
   delete pfes;
   delete vfes;

   delete pfec;
   delete vfec;

   for(int i=1;i<meshes.Size();i++){
       delete meshes[i];
   }

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


   ConstantCoefficient one(1.0);

   //assemble block 12
   bf12.reset(new ParMixedBilinearForm(pfes, vfes));
   //bf12->AddDomainIntegrator(new GradientIntegrator());
   bf12->AddDomainIntegrator(
               new TransposeIntegrator(
                   new VectorDivergenceIntegrator(one)));
   bf12->Assemble(0);
   bf12->Finalize(0);
   A12.reset(bf12->ParallelAssemble());

   //assemble block 21
   bf21.reset(new ParMixedBilinearForm(vfes, pfes));
   bf21->AddDomainIntegrator(
               new VectorDivergenceIntegrator(one));
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

   /*
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
                                           A11.get(),A12.get(),A21.get(),gmres_press);

      prec.reset(lsc);
      prec->SetMaxIter(100);
      prec->SetAbsTol(1e-12);
      prec->SetRelTol(1e-5);
      gmres->SetPreconditioner(*prec);

      ls.reset(gmres);
   }*/

}


void StokesSolver::FSolve()
{
    if(2==sol_method){
        FSolve1();
    }else{
        FSolve2();
    }
}


void StokesSolver::FSolve2()
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


    // use FGMRES for the solver
    std::unique_ptr<mfem::FGMRESSolver> gm;
    gm.reset(new mfem::FGMRESSolver(pfes->GetComm()));
    gm->SetKDim(100);
    gm->SetRelTol(linear_rtol);
    gm->SetAbsTol(linear_atol);
    gm->SetMaxIter(linear_iter);
    gm->SetOperator(*bop);
    gm->SetPrintLevel(1);

    std::unique_ptr<StokesLSCPrec> prec;
    prec.reset(new StokesLSCPrec(vfes,pfes,visc,brink,ess_tdofv,
                                 A11.get(),A12.get(),A21.get()));
    prec->SetMaxIter(50);
    prec->SetAbsTol(1e-15);
    prec->SetRelTol(1e-12);
    prec->iterative_mode=true;

    gm->SetPreconditioner(*prec);

    //solve the linear system
    gm->Mult(rhs,sol);

}


void StokesSolver::FSolve1()
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

   {

       //Solve A11 z = f

       //allocate the preconditioner
       std::unique_ptr<VelocityPrec> prec;
       prec.reset(new VelocityPrec(vfes,visc,brink,ess_tdofv,
                                   A11.get(),A12.get(),A21.get()));

       prec->SetMaxIter(6);
       prec->SetRelTol(1e-12);
       prec->SetAbsTol(1e-12);
       prec->iterative_mode=true;

       std::unique_ptr<mfem::FGMRESSolver> cg11;
       cg11.reset(new mfem::FGMRESSolver(vfes->GetComm()));
       cg11->SetOperator(*A11);
       cg11->SetPreconditioner(*prec);
       cg11->SetAbsTol(linear_atol);
       cg11->SetMaxIter(linear_iter);
       cg11->SetRelTol(linear_rtol);
       cg11->SetPrintLevel(1);
       cg11->iterative_mode=true;
       cg11->Mult(vrhs,vsol);


       // solve the Schur complement system for pressure
       //modify the RHS for pressure
       //prhs=prsh-A21*invA11*f
       Vector tp; tp.SetSize(psol.Size());
       A21->Mult(vsol,tp);
       prhs.Add(-1.0,tp);

       // set the A11 solver for the Schur complement
       cg11->SetPrintLevel(0);
       cg11->iterative_mode=false;
       SchurComplement sc(cg11.get(),A12.get(),A21.get());

       // set the preconditioner for the Schur complement
       SchurComplementLSCPrec scp(vfes,pfes,visc,brink,ess_tdofv,
                                  A11.get(),A12.get(),A21.get());

       //SchurComplementPrec1 scp(vfes,pfes,visc,brink,ess_tdofv,
       //                          A11.get(),A12.get(),A21.get());
       scp.SetAbsTol(linear_atol);
       scp.SetRelTol(linear_rtol);
       scp.SetMaxIter(30);
       scp.SetPrintLevel(-1);
       scp.iterative_mode=true;

       // use FGMRES for the solver
       std::unique_ptr<mfem::FGMRESSolver> gm;
       gm.reset(new mfem::FGMRESSolver(pfes->GetComm()));
       gm->SetOperator(sc);
       gm->SetPreconditioner(scp);
       gm->SetAbsTol(linear_atol);
       gm->SetMaxIter(linear_iter);
       gm->SetRelTol(linear_rtol);
       gm->SetPrintLevel(1);
       gm->SetKDim(50); //Krylov space dimensions

       gm->SetMaxIter(5);
       gm->Mult(prhs,psol); psol.Neg();

       //solve for the velocity with given pressure
       //vrsh=vrsh+A12*psol
       Vector tv; tv.SetSize(vsol.Size());
       A12->Mult(psol,tv);
       vrhs.Add(-1.0,tv);
       cg11->SetPrintLevel(1);
       cg11->iterative_mode=true;
       cg11->Mult(vrhs,vsol);
   }
}

void StokesSolver::FSolve3()
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

    {
        //alocate MG solvers
        for(int i=0;i<meshes.Size()-1;i++){
            StokesSolver* ssolv=new StokesSolver(meshes[i],2);
            for(auto it=vel_bcs.begin();it!=vel_bcs.end();it++){
                ssolv->SetZeroVelocityBC(it->first);
            }
            //assemble the operators
            ssolv->Assemble();
            solvers.Append(ssolv);
            operators.Append(ssolv->GetStokesOperator());
        }
        solvers.Appens(this);
        operators.Append(this->GetStokesOperator());

        //set the prolongations as diagonal block operators
        for(int i=0;i<meshes.Size()-1;i++){
            BlockOperator* pbop=
                    new BlockOperator(solvers[i]->GetTrueBlockOffset(),
                                      solvers[i+1]->GetTrueBlockOffset());
            pbop->owns_blocks=true;
            //set block 0
            pbop->SetDiagonalBlock(0, new TransferOperator(solvers[i]->GetVelociySpace(),
                                                           solvers[i+1]->GetVelocitySpace()));
            pbop->SetDiagonalBlock(1, new TransferOperator(solvers[i]->GetPressureSpace(),
                                                           solvers[i+1]->GetPressureSpace()));


            prolongations.Append(pbop);
        }


    }

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
