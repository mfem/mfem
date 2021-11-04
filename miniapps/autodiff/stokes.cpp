#include "stokes.hpp"
#include "petsc.h"

namespace mfem {

void StokesSolver::FSolve()
{
    tmv=0.0;
    // Set the BC
    ess_tdofv.DeleteAll();
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;
    Array<int> ess_tdofp;
    int dim=pmesh->Dimension();
    {
        for(auto it=bccx.begin();it!=bccx.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);
            ess_tdofx.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(0, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsx from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofx[ii]]=rhs.GetBlock(0)[ess_tdofx[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofx);

        for(auto it=bccy.begin();it!=bccy.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,1);
            ess_tdofy.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(1, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofy[ii]]=rhs.GetBlock(0)[ess_tdofy[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofy);

        for(auto it=bccz.begin();it!=bccz.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,2);
            ess_tdofz.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);

            pcoeff.Set(2, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsz from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofz.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofz[ii]]=rhs.GetBlock(0)[ess_tdofz[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofz);
    }



    if(nfin==nullptr)
    {
        nfin=new StokesIntegratorTH(viscosity,bpenal,load);
        nf->AddDomainIntegrator(nfin);
        nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
    }else{
        nfin->SetParameters(viscosity,bpenal,load);
    }


    // set the RHS
    nf->Mult(sol,rhs);
    rhs.Neg();

    {
        double rhsnorm=mfem::InnerProduct(pmesh->GetComm(),rhs,rhs);
        if(pmesh->GetMyRank()==0){
            std::cout<<"|rhs|="<<std::sqrt(rhsnorm)<<std::endl;
        }
    }

    delete smfem.blPr;
    delete smfem.invA;
    delete smfem.invS;
    delete smfem.S;
    delete smfem.M;

    mfem::BlockOperator& A=nf->GetGradient(sol);
    smfem.A=&A;

    mfem::HypreParMatrix *MinvBt = nullptr;
    mfem::HypreParVector *Md = nullptr;

    mfem::HypreParMatrix* A00=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,0)));
    mfem::HypreParMatrix* A01=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,1)));
    mfem::HypreParMatrix* A10=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,0)));
    mfem::HypreParMatrix* A11=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,1)));

    mfem::HypreParMatrix* A00elim=A00->EliminateRowsCols(ess_tdofv);
    mfem::HypreParMatrix* A11elim=A11->EliminateRowsCols(ess_tdofp);
    mfem::HypreParMatrix* A01elim=A01->EliminateCols(ess_tdofp); A01->EliminateRows(ess_tdofv);
    mfem::HypreParMatrix* A10elim=A10->EliminateCols(ess_tdofv); A10->EliminateRows(ess_tdofp);

    //form mass matrix

    {
        mfem::ConstantCoefficient one(1.0);
        mfem::ParBilinearForm* bf=new mfem::ParBilinearForm(pfes);
        //bf->SetAssemblyLevel(AssemblyLevel::FULL);
        bf->AddDomainIntegrator(new mfem::MassIntegrator(one));
        //bf->AddDomainIntegrator(new mfem::MassIntegrator(*GetBrinkmanPenal()));
        bf->Assemble();
        bf->Finalize();
        smfem.M=bf->ParallelAssemble();
        delete bf;
    }


    //copy BC to RHS

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        rhs.GetBlock(0)[ess_tdofv[ii]]=tmv.GetBlock(0)[ess_tdofv[ii]];
    }


    Md = new HypreParVector(pmesh->GetComm(), A00->GetGlobalNumRows(), A00->GetRowStarts());
    A00->GetDiag(*Md);
    Md->operator *=(9);
    MinvBt = A10->Transpose();
    MinvBt->InvScaleRows(*Md);
    smfem.S = ParMult(A10, MinvBt);

    //invM = new HypreDiagScale(*A00);
    mfem::HypreBoomerAMG* bamg=new HypreBoomerAMG(*A00);
    bamg->SetSystemsOptions(dim);
    bamg->SetElasticityOptions(vfes);
    bamg->SetPrintLevel(print_level);
    smfem.invA = bamg;

    bamg = new HypreBoomerAMG(*smfem.S);
    bamg->SetPrintLevel(print_level);
    smfem.invS = bamg;


/*
    bamg = new HypreBoomerAMG(*smfem.M);
    bamg->SetPrintLevel(print_level);
    smfem.invS = bamg;
*/



    smfem.block_trueOffsets.SetSize(3);
    smfem.block_trueOffsets[0]=0;
    smfem.block_trueOffsets[1]=sol.GetBlock(0).Size();
    smfem.block_trueOffsets[2]=smfem.block_trueOffsets[1]+sol.GetBlock(1).Size();


    smfem.blPr = new BlockDiagonalPreconditioner(smfem.block_trueOffsets);
    smfem.blPr->SetDiagonalBlock(0, smfem.invA);
    smfem.blPr->SetDiagonalBlock(1, smfem.invS);

    mfem::MINRESSolver psol(pmesh->GetComm());
    //mfem::GMRESSolver  psol(pmesh->GetComm());
    psol.SetAbsTol(abs_tol);
    psol.SetRelTol(rel_tol);
    psol.SetMaxIter(20*max_iter);
    psol.SetPrintLevel(print_level);
    psol.SetOperator(*(smfem.A));
    psol.SetPreconditioner(*(smfem.blPr));


    psol.Mult(rhs, tmv);

    if(pmesh->GetMyRank()==0){
        std::cout<<"Num. steps="<<psol.GetNumIterations()<<" conv. norm="<<psol.GetFinalNorm()<<std::endl;
    }

    mfem::Vector zv(pmesh->Dimension()); zv=0.0;
    mfem::VectorConstantCoefficient zz(zv);
    mfem::ParGridFunction& vel=GetVelocity();
    double l2norm0=vel.ComputeL2Error(zz);

    sol.Add(1.0,tmv);
    vel=GetVelocity();
    double l2norm1=vel.ComputeL2Error(zz);

    //std::cout<<"l2norm0="<<l2norm0<<" l2norm1="<<l2norm1<<std::endl;
    //psol->GetConverged();

    delete MinvBt;
    delete Md;

    delete A00elim;
    delete A11elim;
    delete A10elim;
    delete A01elim;
}

void StokesSolver::ASolve(BlockVector &arhs)
{
    if(smfem.blPr==nullptr){
        MFEM_ABORT("StokesSolve::Adjoint - The forward solver should be called first!!!")
    }

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        adj.GetBlock(0)[ess_tdofv[ii]]=0.0;
        tmv.GetBlock(0)[ess_tdofv[ii]]=0.0;
        arhs.GetBlock(0)[ess_tdofv[ii]]=0.0;
    }

    smfem.A->Mult(adj,rhs);
    for(int ii=0;ii<rhs.Size();ii++){
        rhs[ii]=arhs[ii]-rhs[ii];
    }

    mfem::MINRESSolver psol(pmesh->GetComm());
    psol.SetAbsTol(abs_tol);
    psol.SetRelTol(rel_tol);
    psol.SetMaxIter(20*max_iter);
    psol.SetPrintLevel(print_level);
    psol.SetOperator(*(smfem.A));
    psol.SetPreconditioner(*(smfem.blPr));

    psol.Mult(rhs, tmv);

    mfem::Vector zv(pmesh->Dimension()); zv=0.0;
    mfem::VectorConstantCoefficient zz(zv);
    mfem::ParGridFunction& vel=GetAVelocity();
    double l2norm0=vel.ComputeL2Error(zz);


    adj.Add(1.0,tmv);

    vel=GetAVelocity();
    double l2norm1=vel.ComputeL2Error(zz);
    //std::cout<<"l2norm0="<<l2norm0<<" l2norm1="<<l2norm1<<std::endl;

}

void StokesSolver::FSolveN()
{
    tmv=0.0;
    // Set the BC
    ess_tdofv.DeleteAll();
    Array<int> ess_tdofx;
    Array<int> ess_tdofy;
    Array<int> ess_tdofz;
    Array<int> ess_tdofp;
    int dim=pmesh->Dimension();
    {
        for(auto it=bccx.begin();it!=bccx.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,0);
            ess_tdofx.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(0, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsx from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofx.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofx[ii]]=rhs.GetBlock(0)[ess_tdofx[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofx);

        for(auto it=bccy.begin();it!=bccy.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,1);
            ess_tdofy.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);
            pcoeff.Set(1, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsy from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofy.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofy[ii]]=rhs.GetBlock(0)[ess_tdofy[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofy);

        for(auto it=bccz.begin();it!=bccz.end();it++)
        {
            mfem::Array<int> ess_bdr(pmesh->bdr_attributes.Max());
            ess_bdr=0;
            ess_bdr[it->first -1]=1;
            mfem::Array<int> ess_tdof_list;
            vfes->GetEssentialTrueDofs(ess_bdr,ess_tdof_list,2);
            ess_tdofz.Append(ess_tdof_list);

            mfem::VectorArrayCoefficient pcoeff(dim);

            pcoeff.Set(2, it->second, false);
            fvelocity.ProjectBdrCoefficient(pcoeff, ess_bdr);
        }
        //copy tdofsz from velocity grid function
        {
            fvelocity.GetTrueDofs(rhs.GetBlock(0)); // use the rhs vector as a tmp vector
            for(int ii=0;ii<ess_tdofz.Size();ii++)
            {
                sol.GetBlock(0)[ess_tdofz[ii]]=rhs.GetBlock(0)[ess_tdofz[ii]];
            }
        }
        ess_tdofv.Append(ess_tdofz);
    }



    if(nfin==nullptr)
    {
        nfin=new StokesIntegratorTH(viscosity,bpenal,load);
        nf->AddDomainIntegrator(nfin);
        nf->SetGradientType(mfem::Operator::Type::Hypre_ParCSR);
    }else{
        nfin->SetParameters(viscosity,bpenal,load);
    }


    // set the RHS
    nf->Mult(sol,rhs);
    rhs.Neg();



    mfem::BlockOperator& A=nf->GetGradient(sol);

    mfem::HypreParMatrix* A00=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,0)));
    mfem::HypreParMatrix* A01=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(0,1)));
    mfem::HypreParMatrix* A10=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,0)));
    mfem::HypreParMatrix* A11=static_cast<mfem::HypreParMatrix*>(&(A.GetBlock(1,1)));

    mfem::HypreParMatrix* A00elim=A00->EliminateRowsCols(ess_tdofv);
    mfem::HypreParMatrix* A11elim=A11->EliminateRowsCols(ess_tdofp);
    mfem::HypreParMatrix* A01elim=A01->EliminateCols(ess_tdofp); A01->EliminateRows(ess_tdofv);
    mfem::HypreParMatrix* A10elim=A10->EliminateCols(ess_tdofv); A10->EliminateRows(ess_tdofp);

    //copy BC to RHS

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        rhs.GetBlock(0)[ess_tdofv[ii]]=tmv.GetBlock(0)[ess_tdofv[ii]];
    }

    delete psol;
    delete prec;
    delete pmat;
    pmat= new mfem::PetscParMatrix(pmesh->GetComm(),&A, mfem::Operator::PETSC_MATAIJ);
    //set the local block size of the matrix
    Mat sub;
    MatNestGetSubMat(pmat->operator petsc::Mat(),0,0,&sub);
    MatSetBlockSize(sub,dim);


    // construct the preconditioner
    prec = new mfem::PetscFieldSplitSolver(pmesh->GetComm(),*pmat,"prec_");
    // construct the linear solver
    psol = new mfem::PetscLinearSolver(pmesh->GetComm());

    /*
    {
        std::fstream out("full.mat",std::ios::out);
        pmat->PrintMatlab(out);
        out.close();
    }
    */

    psol->SetOperator(*pmat);
    psol->SetPreconditioner(*prec);
    psol->SetAbsTol(abs_tol);
    psol->SetRelTol(rel_tol);
    psol->SetMaxIter(max_iter);
    psol->SetPrintLevel(print_level);

    psol->Mult(rhs, tmv);

    mfem::Vector zv(pmesh->Dimension()); zv=0.0;
    mfem::VectorConstantCoefficient zz(zv);
    mfem::ParGridFunction& vel=GetVelocity();
    double l2norm0=vel.ComputeL2Error(zz);

    sol.Add(1.0,tmv);
    vel=GetVelocity();
    double l2norm1=vel.ComputeL2Error(zz);

    std::cout<<"l2norm0="<<l2norm0<<" l2norm1="<<l2norm1<<std::endl;
    //psol->GetConverged();

    delete A11elim;
    delete A01elim;
    delete A10elim;
    delete A00elim;

}


void StokesSolver::ASolveN(BlockVector& arhs)
{
    if(pmat==nullptr)
    {
        MFEM_ABORT("StokesSolve::Adjoint - The forward solver should be called first!!!")
    }

    for(int ii=0;ii<ess_tdofv.Size();ii++)
    {
        adj.GetBlock(0)[ess_tdofv[ii]]=0.0;
        tmv.GetBlock(0)[ess_tdofv[ii]]=0.0;
        arhs.GetBlock(0)[ess_tdofv[ii]]=0.0;
    }

    pmat->Mult(adj,rhs);
    for(int ii=0;ii<rhs.Size();ii++){
        rhs[ii]=arhs[ii]-rhs[ii];
    }

    psol->Mult(rhs, tmv);

    mfem::Vector zv(pmesh->Dimension()); zv=0.0;
    mfem::VectorConstantCoefficient zz(zv);
    mfem::ParGridFunction& vel=GetAVelocity();
    double l2norm0=vel.ComputeL2Error(zz);


    adj.Add(1.0,tmv);

    vel=GetAVelocity();
    double l2norm1=vel.ComputeL2Error(zz);
    std::cout<<"l2norm0="<<l2norm0<<" l2norm1="<<l2norm1<<std::endl;

}

void StokesSolver::GradD(Vector &grad)
{
    if(dfes==nullptr)
    {
        MFEM_ABORT("StokesSolve::GradD - The design space in not set!!!")
    }

    //set vector size
    grad.SetSize(dfes->GetTrueVSize());
    grad=0.0;

    fvelocity.SetFromTrueDofs(sol.GetBlock(0));
    avelocity.SetFromTrueDofs(adj.GetBlock(0));

    mfem::ParLinearForm lf(dfes);
    lf.AddDomainIntegrator(new StokesGradIntergrator(fvelocity, avelocity
                                                     ,*(ltopopt.fcoef),
                                                     vfes->GetOrder(0)));
    lf.Assemble();
    lf.ParallelAssemble(grad);
}


double StokesSolver::ModelErrors(mfem::GridFunction &el_errors)
{
    if(efes==nullptr)
    {
        int dim=pmesh->Dimension();
        efec=new mfem::L2_FECollection(0,dim);
        efes=new mfem::ParFiniteElementSpace(pmesh,efec,1);
    }

    el_errors.SetSpace(efes);
    el_errors=0.0;
    fvelocity.SetFromTrueDofs(sol.GetBlock(0));
    avelocity.SetFromTrueDofs(adj.GetBlock(0));

    if(ltopopt.tcoef==nullptr)
    {
        ltopopt.tcoef=new mfem::FluidInterpolationCoefficient(*ltopopt.fcoef);
        ltopopt.tcoef->SetPenalization(100*ltopopt.fcoef->GetPenalization());
        ltopopt.tcoef->SetBeta(100*ltopopt.fcoef->GetBeta());
    }
    ltopopt.tcoef->SetGridFunction(density);

    mfem::ParLinearForm lf(efes);
    lf.AddDomainIntegrator(new StokesModelErrorIntegrator(fvelocity,avelocity,
                                                          *(ltopopt.fcoef),
                                                          *(ltopopt.tcoef)));
    lf.Assemble();
    lf.ParallelAssemble(el_errors);

    double loc_err=0.0;
    double tot_err=0.0;
    for(int i=0;i<el_errors.Size();i++){
        loc_err=loc_err+fabs(el_errors[i]);
    }
    MPI_Allreduce(&loc_err, &tot_err, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
    return tot_err;
}


void StokesSolver::TransferData(StokesSolver *nsolver)
{
    nsolver->SetSolver(rel_tol,abs_tol,max_iter,print_level);
    if(viscosity!=nullptr){
        nsolver->SetViscosity(*viscosity);}
    if(load!=nullptr){
        nsolver->SetVolForces(*load);}
    //copy the boundary conditions
    for(auto it=bccx.begin();it!=bccx.end();it++){
        nsolver->AddVelocityBC(it->first,0,*(it->second));
    }
    for(auto it=bccy.begin();it!=bccy.end();it++){
        nsolver->AddVelocityBC(it->first,1,*(it->second));
    }
    for(auto it=bccz.begin();it!=bccz.end();it++){
        nsolver->AddVelocityBC(it->first,2,*(it->second));
    }

    nsolver->SetDesignParameters(ltopopt.eta,ltopopt.beta,
                                 ltopopt.q, ltopopt.lambda);
    if(GetDesignFES()!=nullptr){
        nsolver->SetDesignSpace(GetDesignFES());
        nsolver->GetDesign().ProjectGridFunction(GetDesign());
    }

    //transfer the TargetBrinkmanCoefficient
    FluidInterpolationCoefficient* tbkm=GetTargetBrinkmanPenal();
    nsolver->SetTargetDesignParameters(tbkm->GetEta(), tbkm->GetBeta(), tbkm->GetQ(),tbkm->GetPenalization());
}

/*
void StokesSolver::DiscretizationErrors(Vector &el_errors)
{

   if(hsolver==nullptr)
   {
       //allocate new Stokes solver with one order higher finite element space
       hsolver=new StokesSolver(pmesh,vfes->GetOrder(0)+1);
       TransferData(hsolver);
   }

   if(bpenal!=nullptr){
       hsolver->SetBrinkmanPenal(*(GetBrinkmanPenal()));
   }
   //transfer the solution to hsolver
   hsolver->fvelocity.ProjectGridFunction(fvelocity);
   hsolver->fpressure.ProjectGridFunction(fpressure);
   //Initialize the solution vector
   hsolver->fvelocity.GetTrueDofs(hsolver->sol.GetBlock(0));
   hsolver->fpressure.GetTrueDofs(hsolver->sol.GetBlock(1));
   hsolver->FSolve();
}
*/




}
